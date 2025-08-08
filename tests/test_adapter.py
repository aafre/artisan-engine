"""
Tests for the LlamaCppAdapter class.

These tests verify the model adapter functionality without requiring
actual model files for most tests.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

from artisan_engine.adapter import LlamaCppAdapter
from artisan_engine.exceptions import (
    GenerationError,
    ModelNotFoundError,
    ModelNotLoadedError,
)


class TestUser(BaseModel):
    """Test schema for validation."""

    name: str = Field(description="User's name")
    age: int = Field(description="User's age")


class TestLlamaCppAdapter:
    """Test suite for LlamaCppAdapter."""

    def test_init_with_lazy_loading(self):
        """Test adapter initialization with lazy loading."""
        adapter = LlamaCppAdapter(lazy_loading=True)

        assert adapter.lazy_loading is True
        assert adapter.n_ctx == 2048  # Default value
        assert adapter.is_loaded() is False
        assert adapter._llama_model is None
        assert adapter._outlines_model is None

    def test_init_without_model_path(self):
        """Test adapter initialization without model path."""
        adapter = LlamaCppAdapter()

        assert adapter.model_path is None
        assert adapter.lazy_loading is True
        assert adapter.n_ctx == 2048  # Default value

    def test_init_with_custom_n_ctx(self):
        """Test adapter initialization with custom context size."""
        adapter = LlamaCppAdapter(n_ctx=4096)

        assert adapter.n_ctx == 4096
        assert adapter.llama_kwargs["n_ctx"] == 4096

    def test_init_n_ctx_with_additional_kwargs(self):
        """Test n_ctx with additional llama_cpp parameters."""
        adapter = LlamaCppAdapter(n_ctx=4096, n_gpu_layers=10, verbose=False)

        assert adapter.n_ctx == 4096
        assert adapter.llama_kwargs["n_ctx"] == 4096
        assert adapter.llama_kwargs["n_gpu_layers"] == 10
        assert adapter.llama_kwargs["verbose"] is False

    def test_init_with_string_path(self):
        """Test adapter initialization with string path."""
        test_path = "/test/model.gguf"
        adapter = LlamaCppAdapter(model_path=test_path)

        assert adapter.model_path == Path(test_path)

    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file."""
        adapter = LlamaCppAdapter(model_path="nonexistent.gguf")

        with pytest.raises(ModelNotFoundError):
            adapter.load_model()

    def test_load_model_no_path(self):
        """Test model loading without path."""
        adapter = LlamaCppAdapter()

        with pytest.raises(ModelNotFoundError):
            adapter.load_model()

    @patch("artisan_engine.adapter.outlines.from_llamacpp")
    @patch("artisan_engine.adapter.Llama")
    @patch("pathlib.Path.exists")
    def test_load_model_success(self, mock_exists, mock_llama, mock_outlines):
        """Test successful model loading."""
        # Setup mocks
        mock_exists.return_value = True
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance
        mock_outlines_instance = Mock()
        mock_outlines.return_value = mock_outlines_instance

        # Test
        adapter = LlamaCppAdapter(model_path="test.gguf")
        adapter.load_model()

        # Assertions
        assert adapter.is_loaded() is True
        assert adapter._llama_model == mock_llama_instance
        assert adapter._outlines_model == mock_outlines_instance
        mock_llama.assert_called_once_with(
            "test.gguf", n_ctx=2048
        )  # Now includes n_ctx
        mock_outlines.assert_called_once_with(mock_llama_instance)

    @patch("artisan_engine.adapter.outlines.from_llamacpp")
    @patch("artisan_engine.adapter.Llama")
    @patch("pathlib.Path.exists")
    def test_load_model_exception(self, mock_exists, mock_llama, mock_outlines):
        """Test model loading with exception."""
        mock_exists.return_value = True
        mock_llama.side_effect = Exception("Loading failed")

        adapter = LlamaCppAdapter(model_path="test.gguf")

        with pytest.raises(GenerationError):
            adapter.load_model()

    def test_unload_model(self):
        """Test model unloading."""
        adapter = LlamaCppAdapter()
        adapter._llama_model = Mock()
        adapter._outlines_model = Mock()
        adapter._is_loaded = True
        adapter._generators_cache = {"test": Mock()}

        adapter.unload_model()

        assert adapter._llama_model is None
        assert adapter._outlines_model is None
        assert adapter._is_loaded is False
        assert len(adapter._generators_cache) == 0

    def test_generate_structured_model_not_loaded(self):
        """Test structured generation with unloaded model."""
        adapter = LlamaCppAdapter(lazy_loading=False)

        with pytest.raises(ModelNotLoadedError):
            adapter.generate_structured("test prompt", TestUser)

    @patch("artisan_engine.adapter.outlines.Generator")
    def test_generate_structured_with_lazy_loading(self, mock_generator_class):
        """Test structured generation with lazy loading."""
        # Setup adapter with mocked loaded state
        adapter = LlamaCppAdapter(lazy_loading=True)
        adapter._outlines_model = Mock()
        adapter._is_loaded = True

        # Setup generator mock to return a Pydantic object
        mock_user = TestUser(name="John", age=30)
        mock_generator_instance = Mock()
        mock_generator_instance.return_value = mock_user
        mock_generator_class.return_value = mock_generator_instance

        # Test
        result = adapter.generate_structured(
            "test prompt", TestUser, max_tokens=100, temperature=0.8
        )

        # Assertions
        assert isinstance(result, TestUser)
        assert result.name == "John"
        assert result.age == 30
        mock_generator_class.assert_called_once_with(adapter._outlines_model, TestUser)
        mock_generator_instance.assert_called_once_with(
            "test prompt", max_tokens=100, temperature=0.8
        )

    @patch("artisan_engine.adapter.outlines.Generator")
    def test_generate_structured_cache_generator(self, mock_generator_class):
        """Test generator caching functionality."""
        # Setup
        adapter = LlamaCppAdapter()
        adapter._outlines_model = Mock()
        adapter._is_loaded = True

        mock_user = TestUser(name="John", age=30)
        mock_generator_instance = Mock()
        mock_generator_instance.return_value = mock_user
        mock_generator_class.return_value = mock_generator_instance

        # First call
        result1 = adapter.generate_structured("prompt1", TestUser)

        # Second call with same schema
        result2 = adapter.generate_structured("prompt2", TestUser)

        # Generator should be created only once
        mock_generator_class.assert_called_once()
        assert len(adapter._generators_cache) == 1
        assert "TestUser" in adapter._generators_cache

        # Both results should be TestUser instances
        assert isinstance(result1, TestUser)
        assert isinstance(result2, TestUser)

    def test_health_check_not_loaded(self):
        """Test health check with unloaded model."""
        adapter = LlamaCppAdapter()

        status = adapter.health_check()

        expected = {
            "model_loaded": False,
            "model_path": None,
            "generators_cached": 0,
            "lazy_loading": True,
            "n_ctx": 2048,
        }
        assert status == expected

    def test_health_check_loaded(self):
        """Test health check with loaded model."""
        adapter = LlamaCppAdapter(model_path="test.gguf", n_ctx=4096)
        adapter._is_loaded = True
        adapter._outlines_model = Mock()  # Need this for is_loaded() to return True
        adapter._generators_cache = {"TestUser": Mock(), "Invoice": Mock()}

        # Verify our setup is correct
        assert adapter.is_loaded() is True, "Model should appear loaded for this test"

        status = adapter.health_check()

        expected = {
            "model_loaded": True,
            "model_path": "test.gguf",
            "generators_cached": 2,
            "lazy_loading": True,
            "n_ctx": 4096,
        }
        assert status == expected

    def test_get_model_info_not_loaded(self):
        """Test model info for unloaded model."""
        adapter = LlamaCppAdapter(model_path="test.gguf")

        info = adapter.get_model_info()

        expected = {
            "model_path": "test.gguf",
            "is_loaded": False,
            "lazy_loading": True,
            "n_ctx_configured": 2048,
        }
        assert info == expected

    def test_get_model_info_loaded(self):
        """Test model info for loaded model."""
        adapter = LlamaCppAdapter(model_path="test.gguf", n_ctx=4096)
        adapter._is_loaded = True
        adapter._outlines_model = Mock()  # Need this for is_loaded() to return True

        # Mock llama model with attributes
        mock_llama = Mock()
        mock_llama.n_ctx = 4096  # Actual model context size
        mock_llama.n_vocab = 32000
        adapter._llama_model = mock_llama

        info = adapter.get_model_info()

        assert info["model_path"] == "test.gguf"
        assert info["is_loaded"] is True
        assert info["lazy_loading"] is True
        assert info["n_ctx_configured"] == 4096  # What we configured
        assert info["n_ctx_actual"] == 4096  # What the model actually uses
        assert info["n_vocab"] == 32000
