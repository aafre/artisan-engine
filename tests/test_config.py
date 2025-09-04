"""
Essential configuration tests.

These tests verify critical configuration functionality without over-testing
implementation details.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from artisan_engine.config import (
    Config,
    CORSConfig,
    ModelConfig,
    ServerConfig,
    find_model_file,
    get_model_paths,
)
from artisan_engine.exceptions import ConfigurationError


class TestModelConfig:
    """Test essential ModelConfig functionality."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.path is None
        assert config.lazy_loading is True
        assert config.n_ctx == 2048
        assert config.n_gpu_layers == 0
        assert config.temperature == 0.7
        assert config.max_tokens == 200

    def test_boolean_string_parsing(self):
        """Test boolean string parsing from environment variables."""
        # Test True values
        config = ModelConfig(lazy_loading="true")
        assert config.lazy_loading is True

        config = ModelConfig(lazy_loading="false")
        assert config.lazy_loading is False


class TestServerConfig:
    """Test essential ServerConfig functionality."""

    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.workers == 1
        assert config.reload is False
        assert config.log_level == "info"

    def test_port_validation(self):
        """Test port number validation."""
        # Valid port
        config = ServerConfig(port=8080)
        assert config.port == 8080

        # Invalid ports should raise validation error
        with pytest.raises(ValueError):
            ServerConfig(port=0)

        with pytest.raises(ValueError):
            ServerConfig(port=70000)


class TestCORSConfig:
    """Test essential CORS configuration."""

    def test_default_values(self):
        """Test default CORS configuration."""
        config = CORSConfig()

        assert config.enabled is True
        assert config.allow_origins == ["*"]
        assert config.allow_methods == ["GET", "POST"]
        assert config.allow_headers == ["*"]

    def test_comma_separated_parsing(self):
        """Test parsing comma-separated strings."""
        config = CORSConfig(
            allow_origins="http://localhost:3000,https://example.com"
        )

        assert config.allow_origins == ["http://localhost:3000", "https://example.com"]


class TestMainConfig:
    """Test main Config class functionality."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        with patch.dict(os.environ, {
            "ARTISAN_REQUIRE_MODEL": "false",
            "ARTISAN_DEBUG": "false"  # Explicitly set debug to false
        }, clear=True):
            config = Config()

            assert config.environment == "development"
            assert config.debug is False
            assert config.version == "0.1.0"
            assert isinstance(config.model, ModelConfig)
            assert isinstance(config.server, ServerConfig)
            assert isinstance(config.cors, CORSConfig)

    @patch.dict(os.environ, {
        "ARTISAN_MODEL_PATH": "/test/model.gguf",
        "ARTISAN_SERVER_PORT": "9000",
        "ARTISAN_DEBUG": "true",
        "ARTISAN_REQUIRE_MODEL": "false"
    })
    def test_environment_variable_parsing(self):
        """Test parsing key environment variables."""
        config = Config()

        assert config.model.path == Path("/test/model.gguf")
        assert config.server.port == 9000
        assert config.debug is True


class TestModelPathDiscovery:
    """Test model file discovery functionality."""

    @patch("glob.glob")
    @patch("artisan_engine.config.get_config")
    def test_get_model_paths_basic(self, mock_get_config, mock_glob):
        """Test basic model paths functionality."""
        mock_config = Config()
        mock_config.model.path = Path("/custom/model.gguf")
        mock_get_config.return_value = mock_config
        mock_glob.return_value = []

        paths = get_model_paths()

        assert Path("/custom/model.gguf") in paths

    def test_find_model_file_basic(self):
        """Test model file discovery basic functionality."""
        # Test that function exists and returns expected type
        result = find_model_file()
        assert result is None or isinstance(result, Path)

    @patch("artisan_engine.config.get_model_paths")
    def test_find_model_file_none_found(self, mock_get_paths):
        """Test model file discovery when no files exist."""
        mock_get_paths.return_value = []

        found_path = find_model_file()
        assert found_path is None


class TestConfigValidation:
    """Test critical configuration validation."""

    def test_validate_invalid_port(self):
        """Test validation failure for invalid port."""
        with pytest.raises(ConfigurationError, match="Invalid port number"):
            from artisan_engine.config import _validate_config
            config = Config()
            config.server.port = 70000
            _validate_config(config)

    def test_validate_invalid_log_level(self):
        """Test validation failure for invalid log level."""
        with pytest.raises(ConfigurationError, match="Invalid log level"):
            from artisan_engine.config import _validate_config
            config = Config()
            config.server.log_level = "invalid_level"
            _validate_config(config)
