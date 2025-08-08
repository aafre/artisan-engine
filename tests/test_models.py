"""
Tests for Pydantic models and schemas.

These tests verify that our API models work correctly and validate
data as expected.
"""

import pytest
from pydantic import ValidationError

from artisan_engine.models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthRequest,
    HealthResponse,
    OpenAIChatRequest,
    OpenAIMessage,
    ResponseFormat,
    ResponseFormatJsonSchema,
)
from artisan_engine.schemas import Invoice, LineItem, User, get_schema, list_schemas


class TestExampleSchemas:
    """Test the example schemas from the POC."""

    def test_user_schema_valid(self):
        """Test valid User schema creation."""
        user = User(
            name="John Doe",
            age=30,
            is_subscribed=True,
            interests=["programming", "hiking"],
        )

        assert user.name == "John Doe"
        assert user.age == 30
        assert user.is_subscribed is True
        assert user.interests == ["programming", "hiking"]

    def test_user_schema_invalid(self):
        """Test User schema validation failures."""
        # Missing required field
        with pytest.raises(ValidationError):
            User(name="John", age=30, is_subscribed=True)  # Missing interests

        # Wrong type
        with pytest.raises(ValidationError):
            User(name="John", age="thirty", is_subscribed=True, interests=[])

        # Wrong interest type
        with pytest.raises(ValidationError):
            User(name="John", age=30, is_subscribed=True, interests="programming")

    def test_line_item_schema(self):
        """Test LineItem schema."""
        item = LineItem(item="Widget", quantity=2, price=19.99)

        assert item.item == "Widget"
        assert item.quantity == 2
        assert item.price == 19.99

    def test_invoice_schema_valid(self):
        """Test valid Invoice schema."""
        items = [
            LineItem(item="Widget", quantity=2, price=19.99),
            LineItem(item="Gadget", quantity=1, price=9.99),
        ]

        invoice = Invoice(
            invoice_id="INV-001",
            customer_name="ACME Corp",
            items=items,
            tax_amount=5.99,
        )

        assert invoice.invoice_id == "INV-001"
        assert invoice.customer_name == "ACME Corp"
        assert len(invoice.items) == 2
        assert invoice.tax_amount == 5.99

    def test_invoice_schema_optional_tax(self):
        """Test Invoice schema with optional tax amount."""
        items = [LineItem(item="Widget", quantity=1, price=10.0)]

        invoice = Invoice(invoice_id="INV-002", customer_name="Test Corp", items=items)

        assert invoice.tax_amount is None


class TestRequestModels:
    """Test API request models."""

    def test_generate_request_valid(self):
        """Test valid GenerateRequest."""
        request = GenerateRequest(
            prompt="Generate a user",
            schema_name="User",
            max_tokens=150,
            temperature=0.8,
        )

        assert request.prompt == "Generate a user"
        assert request.schema_name == "User"
        assert request.max_tokens == 150
        assert request.temperature == 0.8

    def test_generate_request_defaults(self):
        """Test GenerateRequest with default values."""
        request = GenerateRequest(prompt="Test prompt", schema_name="User")

        assert request.max_tokens == 200  # Default
        assert request.temperature == 0.7  # Default
        assert request.extra_params is None  # Default

    def test_generate_request_validation(self):
        """Test GenerateRequest validation."""
        # Empty prompt
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="", schema_name="User")

        # Temperature out of range
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", schema_name="User", temperature=2.0)

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", schema_name="User", temperature=-0.1)

        # Invalid max_tokens
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", schema_name="User", max_tokens=0)

    def test_openai_message(self):
        """Test OpenAI message model."""
        message = OpenAIMessage(role="user", content="Hello")

        assert message.role == "user"
        assert message.content == "Hello"

    def test_response_format_models(self):
        """Test OpenAI response_format models."""
        # Test JSON schema
        json_schema = ResponseFormatJsonSchema(
            properties={"name": {"type": "string"}, "age": {"type": "integer"}},
            required=["name", "age"],
        )

        assert json_schema.type == "object"  # Default
        assert json_schema.properties["name"]["type"] == "string"
        assert json_schema.required == ["name", "age"]

        # Test response format
        response_format = ResponseFormat(type="json_object", json_schema=json_schema)

        assert response_format.type == "json_object"
        assert response_format.json_schema == json_schema

    def test_openai_chat_request(self):
        """Test OpenAI chat request model with response_format."""
        messages = [
            OpenAIMessage(role="user", content="Hello"),
            OpenAIMessage(role="assistant", content="Hi there!"),
        ]

        json_schema = ResponseFormatJsonSchema(
            properties={"response": {"type": "string"}}, required=["response"]
        )

        response_format = ResponseFormat(type="json_object", json_schema=json_schema)

        request = OpenAIChatRequest(
            model="gpt-4",
            messages=messages,
            response_format=response_format,
            max_tokens=100,
            temperature=0.5,
        )

        assert request.model == "gpt-4"
        assert len(request.messages) == 2
        assert request.response_format.type == "json_object"
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is False  # Default

    def test_health_request(self):
        """Test health request model."""
        # With defaults
        request = HealthRequest()
        assert request.include_details is False

        # With details
        request = HealthRequest(include_details=True)
        assert request.include_details is True


class TestResponseModels:
    """Test API response models."""

    def test_generate_response(self):
        """Test GenerateResponse model."""
        response = GenerateResponse(
            success=True,
            result='{"name": "John", "age": 30}',
            parsed_result={"name": "John", "age": 30},
            schema_name="User",
            generation_time=1.5,
            token_count=20,
        )

        assert response.success is True
        assert response.result == '{"name": "John", "age": 30}'
        assert response.parsed_result == {"name": "John", "age": 30}
        assert response.schema_name == "User"
        assert response.generation_time == 1.5
        assert response.token_count == 20

    def test_health_response(self):
        """Test HealthResponse model."""
        response = HealthResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00",
            version="0.1.0",
            model_status=None,
            uptime=3600.0,
        )

        assert response.status == "healthy"
        assert response.timestamp == "2024-01-01T00:00:00"
        assert response.version == "0.1.0"
        assert response.model_status is None
        assert response.uptime == 3600.0

    def test_error_response(self):
        """Test ErrorResponse model."""
        response = ErrorResponse(
            error="Something went wrong",
            error_code="TEST_ERROR",
            timestamp="2024-01-01T00:00:00",
            details={"extra": "info"},
        )

        assert response.success is False  # Always False
        assert response.error == "Something went wrong"
        assert response.error_code == "TEST_ERROR"
        assert response.timestamp == "2024-01-01T00:00:00"
        assert response.details == {"extra": "info"}


class TestSchemaRegistry:
    """Test schema registry functionality."""

    def test_get_schema_valid(self):
        """Test getting valid schemas."""
        user_schema = get_schema("User")
        assert user_schema == User

        invoice_schema = get_schema("Invoice")
        assert invoice_schema == Invoice

        line_item_schema = get_schema("LineItem")
        assert line_item_schema == LineItem

    def test_get_schema_invalid(self):
        """Test getting invalid schema."""
        schema = get_schema("NonexistentSchema")
        assert schema is None

    def test_list_schemas(self):
        """Test listing available schemas."""
        schemas = list_schemas()

        assert isinstance(schemas, list)
        assert "User" in schemas
        assert "Invoice" in schemas
        assert "LineItem" in schemas
        assert len(schemas) >= 3


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_user_json_serialization(self):
        """Test User model JSON serialization."""
        user = User(
            name="Jane Doe",
            age=25,
            is_subscribed=False,
            interests=["reading", "travel"],
        )

        # Test model_dump_json
        json_str = user.model_dump_json()
        assert isinstance(json_str, str)
        assert "Jane Doe" in json_str
        assert "reading" in json_str

        # Test model_validate_json
        user_restored = User.model_validate_json(json_str)
        assert user_restored == user

    def test_generate_request_serialization(self):
        """Test GenerateRequest serialization."""
        request = GenerateRequest(
            prompt="Test prompt",
            schema_name="User",
            max_tokens=100,
            temperature=0.5,
            extra_params={"custom": "value"},
        )

        # Test model_dump
        data = request.model_dump()
        assert data["prompt"] == "Test prompt"
        assert data["schema_name"] == "User"
        assert data["extra_params"]["custom"] == "value"

        # Test reconstruction
        request_restored = GenerateRequest(**data)
        assert request_restored.prompt == request.prompt
        assert request_restored.extra_params == request.extra_params

    def test_complex_invoice_serialization(self):
        """Test complex nested model serialization."""
        items = [
            LineItem(item="Laptop", quantity=1, price=999.99),
            LineItem(item="Mouse", quantity=2, price=29.99),
        ]

        invoice = Invoice(
            invoice_id="INV-12345",
            customer_name="Tech Corp",
            items=items,
            tax_amount=106.00,
        )

        # Serialize to JSON
        json_data = invoice.model_dump_json()

        # Deserialize back
        invoice_restored = Invoice.model_validate_json(json_data)

        assert invoice_restored.invoice_id == invoice.invoice_id
        assert invoice_restored.customer_name == invoice.customer_name
        assert len(invoice_restored.items) == 2
        assert invoice_restored.items[0].item == "Laptop"
        assert invoice_restored.items[1].quantity == 2
        assert invoice_restored.tax_amount == 106.00
