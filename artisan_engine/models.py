"""
Pydantic models for API request/response handling.

This module defines all the data models used for API requests and responses,
ensuring proper validation and serialization.
"""

from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# API REQUEST MODELS
# =============================================================================


class GenerateRequest(BaseModel):
    """Request model for structured generation."""

    prompt: str = Field(
        description="The input prompt for generation", min_length=1, max_length=10000
    )
    schema_name: str = Field(
        description="Name of the schema to use for structured output",
        examples=["User", "Invoice"],
    )
    max_tokens: int = Field(
        default=200, description="Maximum number of tokens to generate", ge=1, le=4096
    )
    temperature: float = Field(
        default=0.7, description="Generation temperature (0.0-1.0)", ge=0.0, le=1.0
    )
    extra_params: dict[str, Any] | None = Field(
        default=None, description="Additional generation parameters"
    )


# =============================================================================
# OPENAI COMPATIBILITY MODELS
# =============================================================================


class OpenAIMessage(BaseModel):
    """OpenAI-compatible message format."""

    role: str = Field(description="Message role (user, assistant, system)")
    content: str = Field(description="Message content")


class ResponseFormatJsonSchema(BaseModel):
    """The JSON schema definition for structured output."""

    type: str = Field(default="object", description="Schema type")
    properties: dict[str, Any] = Field(description="Schema properties")
    required: list[str] = Field(default_factory=list, description="Required fields")
    additionalProperties: bool = Field(
        default=False, description="Allow additional properties"
    )


class ResponseFormat(BaseModel):
    """The response_format object for OpenAI structured output compatibility."""

    type: str = Field(
        description="Response format type (json_object, json_schema, text)"
    )
    json_schema: ResponseFormatJsonSchema | None = Field(
        default=None, description="JSON schema definition for structured output"
    )

    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request matching the official API."""

    model: str = Field(description="Model identifier")
    messages: list[OpenAIMessage] = Field(description="Conversation messages")
    response_format: ResponseFormat | None = Field(
        default=None,
        description="An object specifying the format that the model must output",
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate"
    )
    temperature: float | None = Field(
        default=1.0, ge=0.0, le=2.0, description="Generation temperature"
    )
    top_p: float | None = Field(
        default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    frequency_penalty: float | None = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: float | None = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty"
    )
    stop: str | list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool | None = Field(default=False, description="Stream response")
    user: str | None = Field(default=None, description="User identifier")

    class Config:
        # Allow extra fields for forward compatibility with OpenAI API
        extra = "allow"


class HealthRequest(BaseModel):
    """Health check request (typically empty)."""

    include_details: bool = Field(
        default=False, description="Include detailed health information"
    )


# =============================================================================
# API RESPONSE MODELS
# =============================================================================


class GenerateResponse(BaseModel):
    """Response model for structured generation."""

    success: bool = Field(description="Whether generation succeeded")
    result: str | None = Field(default=None, description="Generated JSON string")
    parsed_result: dict[str, Any] | None = Field(
        default=None, description="Parsed and validated result object"
    )
    schema_name: str = Field(description="Schema used for generation")
    generation_time: float = Field(description="Time taken for generation (seconds)")
    token_count: int | None = Field(
        default=None, description="Number of tokens generated"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata"
    )


class OpenAIChoice(BaseModel):
    """OpenAI-compatible choice object."""

    index: int
    message: OpenAIMessage
    finish_reason: str


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: dict[str, int] | None = None


class ModelInfo(BaseModel):
    """Information about a model."""

    id: str = Field(description="Model identifier")
    name: str = Field(description="Human-readable model name")
    path: str | None = Field(description="Model file path")
    loaded: bool = Field(description="Whether model is loaded")
    parameters: dict[str, Any] | None = Field(description="Model parameters")


class ModelsResponse(BaseModel):
    """Response for available models endpoint."""

    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Health status (healthy, unhealthy)")
    timestamp: str = Field(description="Response timestamp")
    version: str | None = Field(description="API version")
    model_status: dict[str, Any] | None = Field(
        description="Detailed model status information"
    )
    uptime: float | None = Field(description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Structured error response."""

    success: bool = False
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code identifier")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    timestamp: str = Field(description="Error timestamp")
