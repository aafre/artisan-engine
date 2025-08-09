"""
Tests for the FastAPI application endpoints.

These tests verify API functionality using FastAPI's test client.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from artisan_engine.schemas import User
from main import app, get_adapter, get_adapter_optional


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_adapter():
    """Create mock adapter using FastAPI dependency override."""
    mock_instance = Mock()

    # Set up the default return values for the mock's methods
    # This ensures that any test using the fixture gets a pre-configured mock.
    mock_instance.health_check.return_value = {
        "model_loaded": True,
        "model_path": "mocked.gguf",
    }
    mock_instance.get_model_info.return_value = {
        "model_path": "mocked.gguf",
        "is_loaded": True,
    }
    # Set a default return for the main generation method too
    mock_instance.generate_structured.return_value = User(
        name="Mocked User", age=99, is_subscribed=True, interests=[]
    )

    # Override FastAPI dependencies
    app.dependency_overrides[get_adapter] = lambda: mock_instance
    app.dependency_overrides[get_adapter_optional] = lambda: mock_instance

    yield mock_instance

    # Clear dependency overrides after the test suite runs
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_basic(self, client, mock_adapter):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data

    def test_health_with_details(self, client, mock_adapter):
        """Test health check with details."""
        response = client.get("/health", params={"include_details": True})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_status"] is not None
        assert data["model_status"]["model_loaded"] is True

    @patch("main.adapter", None)
    def test_health_no_adapter(self, client):
        """Test health check with no adapter."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


class TestModelsEndpoint:
    """Test models listing endpoint."""

    def test_list_models(self, client, mock_adapter):
        """Test listing available models."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 0

        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert "name" in model
            assert "loaded" in model

    @patch("main.adapter", None)
    def test_list_models_no_adapter(self, client):
        """Test listing models with no adapter."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 0


class TestSchemasEndpoint:
    """Test schemas listing endpoint."""

    def test_list_schemas(self, client):
        """Test listing available schemas."""
        response = client.get("/schemas")

        assert response.status_code == 200
        data = response.json()
        assert "registered_schemas" in data
        assert "registered_count" in data
        assert isinstance(data["registered_schemas"], list)
        assert data["registered_count"] >= 0
        assert "dynamic_schemas_cached" in data
        assert "total_schemas_available" in data


class TestGenerateEndpoint:
    """Test structured generation endpoint."""

    def test_generate_success(self, client, mock_adapter):
        """Test successful generation."""
        # Mock the generation response to return a Pydantic object
        mock_user = User(name="John Doe", age=30, is_subscribed=True, interests=["AI"])
        mock_adapter.generate_structured.return_value = mock_user

        request_data = {
            "prompt": "Generate a user named John Doe, age 30",
            "schema_name": "User",
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = client.post("/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "John Doe" in data["result"]  # JSON string should contain the name
        assert data["schema_name"] == "User"
        assert "generation_time" in data
        assert data["parsed_result"] is not None
        assert data["parsed_result"]["name"] == "John Doe"

    def test_generate_unknown_schema(self, client, mock_adapter):
        """Test generation with unknown schema."""
        request_data = {"prompt": "Test prompt", "schema_name": "UnknownSchema"}

        response = client.post("/generate", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "Unknown schema" in data["error"]

    def test_generate_empty_prompt(self, client, mock_adapter):
        """Test generation with empty prompt."""
        request_data = {"prompt": "", "schema_name": "User"}

        response = client.post("/generate", json=request_data)

        # Should fail validation due to min_length constraint
        assert response.status_code == 422

    @patch("main.adapter", None)
    def test_generate_no_adapter(self, client):
        """Test generation with no adapter."""
        request_data = {"prompt": "Test prompt", "schema_name": "User"}

        response = client.post("/generate", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "Model adapter not initialized" in data["error"]


class TestChatCompletionsEndpoint:
    """Test OpenAI-compatible chat completions endpoint."""

    def test_chat_completions_success(self, client, mock_adapter):
        """Test successful chat completion with OpenAI-compatible format."""
        mock_user = User(name="Alice", age=25, is_subscribed=False, interests=["music"])
        mock_adapter.generate_structured.return_value = mock_user

        # Use proper OpenAI response_format structure
        request_data = {
            "model": "local-llm",
            "messages": [
                {"role": "user", "content": "Generate a user named Alice, age 25"}
            ],
            "response_format": {
                "type": "json_object",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "is_subscribed": {"type": "boolean"},
                        "interests": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "age", "is_subscribed", "interests"],
                },
            },
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert data["model"] == "local-llm"
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["finish_reason"] == "stop"
        assert "Alice" in choice["message"]["content"]

    def test_chat_completions_no_messages(self, client, mock_adapter):
        """Test chat completion with no messages."""
        request_data = {"model": "local-llm", "messages": [], "schema_name": "User"}

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "No messages provided" in data["error"]

    def test_chat_completions_no_response_format(self, client, mock_adapter):
        """Test chat completion without response_format."""
        request_data = {
            "model": "local-llm",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "requires structured output" in data["error"]


class TestRootEndpoint:
    """Test root information endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Artisan Engine"
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
        assert "documentation" in data

        # Check that key endpoints are listed
        endpoints = data["endpoints"]
        assert "/health" in endpoints.values()
        assert "/models" in endpoints.values()
        assert "/generate" in endpoints.values()


class TestErrorHandling:
    """Test error handling."""

    def test_404_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_generation_error_handling(self, client, mock_adapter):
        """Test handling of generation errors."""
        # Mock adapter to raise an exception
        mock_adapter.generate_structured.side_effect = Exception("Generation failed")

        request_data = {"prompt": "Test prompt", "schema_name": "User"}

        response = client.post("/generate", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "Generation failed" in data["error"]
        assert data["error_code"] == "GENERATION_ERROR"


class TestRequestValidation:
    """Test request validation."""

    def test_generate_request_validation(self, client, mock_adapter):
        """Test validation of generate request."""
        # Missing required fields
        response = client.post("/generate", json={})
        assert response.status_code == 422

        # Invalid temperature
        request_data = {
            "prompt": "Test",
            "schema_name": "User",
            "temperature": 2.0,  # Out of range
        }
        response = client.post("/generate", json=request_data)
        assert response.status_code == 422

        # Invalid max_tokens
        request_data = {
            "prompt": "Test",
            "schema_name": "User",
            "max_tokens": -1,  # Negative
        }
        response = client.post("/generate", json=request_data)
        assert response.status_code == 422

    
    
def test_get_version(client):
    response = client.get("/version")
    assert response.status_code == 200
    json_data = response.json()
    assert "version" in json_data
    assert isinstance(json_data["version"], str)



