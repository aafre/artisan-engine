"""
Configuration management for the artisan-engine API.

This module handles environment-based configuration, providing defaults
and validation for all configuration options.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Configuration for model settings."""

    path: Path | None = Field(default=None, description="Path to model file")
    lazy_loading: bool = Field(default=True, description="Enable lazy model loading")
    n_ctx: int = Field(default=2048, description="Context window size")
    n_gpu_layers: int = Field(default=0, description="Number of GPU layers")
    temperature: float = Field(default=0.7, description="Default temperature")
    max_tokens: int = Field(default=200, description="Default max tokens")

    @field_validator("lazy_loading", mode="before")
    @classmethod
    def parse_boolean_string(cls, v: Any) -> bool:
        """Parse boolean strings from environment variables."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    @model_validator(mode="after")
    def find_model_path_if_not_set(self) -> "ModelConfig":
        """Find model path from default locations if not explicitly set."""
        # Only search for default path if no environment variable was set
        if self.path is None and not os.getenv("ARTISAN_MODEL_PATH"):
            # Default model paths to check
            default_paths = [
                "./models/*.gguf",  # Docker mount point
                "./local_llms/*.gguf",  # Development directory
                "./Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
                "./Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            ]
            for path_pattern in default_paths:
                if "*" in str(path_pattern):
                    # Handle glob patterns
                    matches = glob.glob(str(path_pattern))
                    if matches:
                        self.path = Path(matches[0])  # Use first match
                        break
                else:
                    if Path(path_pattern).exists():
                        self.path = Path(path_pattern)
                        break
        return self


class ServerConfig(BaseModel):
    """Configuration for API server settings."""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port", ge=1, le=65535)
    workers: int = Field(default=1, description="Number of workers", ge=1)
    reload: bool = Field(default=False, description="Enable auto-reload")
    log_level: str = Field(default="info", description="Logging level")

    @field_validator("reload", mode="before")
    @classmethod
    def parse_boolean_string(cls, v: Any) -> bool:
        """Parse boolean strings from environment variables."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v


class CORSConfig(BaseModel):
    """Configuration for CORS settings."""

    enabled: bool = Field(default=True, description="Enable CORS")
    allow_origins: list[str] = Field(default=["*"], description="Allowed origins")
    allow_methods: list[str] = Field(
        default=["GET", "POST"], description="Allowed methods"
    )
    allow_headers: list[str] = Field(default=["*"], description="Allowed headers")

    @field_validator("enabled", mode="before")
    @classmethod
    def parse_boolean_string(cls, v: Any) -> bool:
        """Parse boolean strings from environment variables."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    @field_validator("allow_origins", "allow_methods", "allow_headers", mode="before")
    @classmethod
    def parse_comma_separated_strings(cls, v: Any) -> list[str]:
        """Parse comma-separated strings from environment variables into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class Config(BaseSettings):
    """Main configuration class using pydantic-settings."""

    # Environment and metadata
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    version: str = Field(default="0.1.0", description="API version")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_boolean(cls, v: Any) -> bool:
        """Parse boolean strings from environment variables."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    @model_validator(mode="after")
    def parse_nested_fields_from_env(self) -> "Config":
        """Parse environment variables for nested models that pydantic-settings misses."""
        # Model configuration
        if model_path_str := os.getenv("ARTISAN_MODEL_PATH"):
            self.model.path = Path(model_path_str) if model_path_str else None
        if lazy_loading_str := os.getenv("ARTISAN_MODEL_LAZY_LOADING"):
            self.model.lazy_loading = lazy_loading_str.lower() in (
                "true",
                "1",
                "yes",
                "on",
            )
        if n_ctx_str := os.getenv("ARTISAN_MODEL_N_CTX"):
            try:
                self.model.n_ctx = int(n_ctx_str)
            except ValueError:
                pass
        if n_gpu_layers_str := os.getenv("ARTISAN_MODEL_N_GPU_LAYERS"):
            try:
                self.model.n_gpu_layers = int(n_gpu_layers_str)
            except ValueError:
                pass
        if temperature_str := os.getenv("ARTISAN_MODEL_TEMPERATURE"):
            try:
                self.model.temperature = float(temperature_str)
            except ValueError:
                pass
        if max_tokens_str := os.getenv("ARTISAN_MODEL_MAX_TOKENS"):
            try:
                self.model.max_tokens = int(max_tokens_str)
            except ValueError:
                pass

        # Server configuration
        if host_str := os.getenv("ARTISAN_SERVER_HOST"):
            self.server.host = host_str
        if port_str := os.getenv("ARTISAN_SERVER_PORT"):
            try:
                self.server.port = int(port_str)
            except ValueError:
                pass
        if workers_str := os.getenv("ARTISAN_SERVER_WORKERS"):
            try:
                self.server.workers = int(workers_str)
            except ValueError:
                pass
        if reload_str := os.getenv("ARTISAN_SERVER_RELOAD"):
            self.server.reload = reload_str.lower() in ("true", "1", "yes", "on")
        if log_level_str := os.getenv("ARTISAN_SERVER_LOG_LEVEL"):
            self.server.log_level = log_level_str

        # CORS configuration
        if cors_enabled_str := os.getenv("ARTISAN_CORS_ENABLED"):
            self.cors.enabled = cors_enabled_str.lower() in ("true", "1", "yes", "on")
        if cors_origins_str := os.getenv("ARTISAN_CORS_ALLOW_ORIGINS"):
            self.cors.allow_origins = [
                item.strip() for item in cors_origins_str.split(",") if item.strip()
            ]
        if cors_methods_str := os.getenv("ARTISAN_CORS_ALLOW_METHODS"):
            self.cors.allow_methods = [
                item.strip() for item in cors_methods_str.split(",") if item.strip()
            ]
        if cors_headers_str := os.getenv("ARTISAN_CORS_ALLOW_HEADERS"):
            self.cors.allow_headers = [
                item.strip() for item in cors_headers_str.split(",") if item.strip()
            ]

        # Main level configuration
        if version_str := os.getenv("ARTISAN_VERSION"):
            self.version = version_str
        if log_format_str := os.getenv("ARTISAN_LOG_FORMAT"):
            self.log_format = log_format_str

        return self

    require_model: bool = Field(
        default=True, description="Require model file to be present at startup"
    )

    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Server configuration
    server: ServerConfig = Field(default_factory=ServerConfig)

    # CORS configuration
    cors: CORSConfig = Field(default_factory=CORSConfig)

    # Logging configuration
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    model_config = SettingsConfigDict(
        env_prefix="ARTISAN_",
        env_nested_delimiter="_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        env_parse_none_str="null",
    )


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_config() -> Config:
    """
    Load configuration from .env files, environment variables and defaults.

    Returns:
        Loaded configuration instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Load .env files (searches current dir and parents)
        # This loads variables into os.environ, which pydantic-settings then reads
        load_dotenv(verbose=False, override=False)

        # Load base configuration (pydantic-settings will automatically read .env)
        config = Config()

        # Validate configuration
        _validate_config(config)

        return config

    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


def _validate_config(config: Config) -> None:
    """
    Validate configuration values.

    Args:
        config: Configuration to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Validate model path if provided
    if config.model.path and not config.model.path.exists():
        raise ConfigurationError(f"Model file not found: {config.model.path}")

    # Validate server configuration
    if not (1 <= config.server.port <= 65535):
        raise ConfigurationError(f"Invalid port number: {config.server.port}")

    # Validate log level
    valid_log_levels = ["debug", "info", "warning", "error", "critical"]
    if config.server.log_level.lower() not in valid_log_levels:
        raise ConfigurationError(f"Invalid log level: {config.server.log_level}")


def setup_logging(config: Config | None = None) -> None:
    """
    Set up logging configuration.

    Args:
        config: Configuration instance (uses global if None)
    """
    if config is None:
        config = get_config()

    log_level = getattr(logging, config.server.log_level.upper())

    logging.basicConfig(
        level=log_level, format=config.log_format, handlers=[logging.StreamHandler()]
    )

    # Set specific loggers
    loggers_to_configure = [
        "artisan_engine",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


def get_model_paths() -> list[Path]:
    """
    Get list of potential model file paths.

    Returns:
        List of paths to check for model files
    """
    config = get_config()
    paths = []

    if config.model.path:
        paths.append(config.model.path)

    # Add common default paths
    import glob

    default_patterns = [
        "./models/*.gguf",  # Docker mount point
        "./local_llms/*.gguf",  # Development directory
        "./Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
        "./Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    ]

    default_paths = []
    for pattern in default_patterns:
        if "*" in str(pattern):
            # Expand glob patterns
            matches = glob.glob(str(pattern))
            default_paths.extend(matches)
        else:
            default_paths.append(pattern)

    for path_str in default_paths:
        path = Path(path_str)
        if path not in paths:
            paths.append(path)

    return paths


def find_model_file() -> Path | None:
    """
    Find the first available model file from configured paths.

    Returns:
        Path to model file if found, None otherwise
    """
    for path in get_model_paths():
        if path.exists():
            return path
    return None
