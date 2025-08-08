"""
Artisan Engine - OpenAI-compatible API for local LLMs.

This FastAPI application provides a robust HTTP interface for structured generation
using local language models with guaranteed JSON output.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi.params import Query
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from artisan_engine.adapter import LlamaCppAdapter
from artisan_engine.config import find_model_file, get_config, setup_logging
from artisan_engine.exceptions import (
    ArtisanEngineError,
    ConfigurationError,
    ModelNotLoadedError,
    ConfigurationError,
    GenerationError,
    ModelNotLoadedError,
    ValidationError,
)
from artisan_engine.models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
)
from artisan_engine.schemas import (
    find_or_create_schema,
    get_schema,
    get_schema_cache_info,
)

# Global adapter instance
adapter: LlamaCppAdapter | None = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global adapter

    logger = logging.getLogger("artisan_engine")

    # Startup
    config = get_config()
    setup_logging(config)

    # Initialize adapter
    model_path = find_model_file()
    if not model_path:
        if config.require_model:
            import sys
            error_msg = (
                "ERROR: No model file found! To run Artisan Engine, you need to provide a model file.\n\n"
                "For Docker containers:\n"
                "  docker run -p 8000:8000 -v /path/to/your/models:/app/models artisan-engine\n"
                "  docker run -p 8000:8000 -e ARTISAN_MODEL_PATH=/app/model.gguf -v /path/to/model.gguf:/app/model.gguf artisan-engine\n\n"
                "For local development:\n"
                "  Set ARTISAN_MODEL_PATH environment variable\n"
                "  Place model files in: ./local_llms/ or current directory\n\n"
                "Supported model files: *.gguf (GGML/llama.cpp format)\n\n"
                "To disable this check, set ARTISAN_REQUIRE_MODEL=false\n"
            )
            raise ConfigurationError(error_msg)
        else:
            # Don't fail startup, allow lazy loading
            adapter = LlamaCppAdapter(lazy_loading=True)
            logger.warning("No model file found - using lazy loading mode")
    else:
        adapter = LlamaCppAdapter(
            model_path=model_path,
            lazy_loading=config.model.lazy_loading,
            n_ctx=config.model.n_ctx,
            n_gpu_layers=config.model.n_gpu_layers,
        )

        # Pre-load if not lazy loading
        if not config.model.lazy_loading:
            try:
                adapter.load_model()
                logger.info("Model pre-loaded successfully")
            except Exception as e:
                logger.error(f"Failed to pre-load model: {e}")

    yield

    # Shutdown
    if adapter:
        adapter.unload_model()


# Initialize FastAPI app
config = get_config()
app = FastAPI(
    title="Artisan Engine",
    description="Production-grade OpenAI-compatible API for local LLMs with guaranteed structured output",
    version=config.version,
    lifespan=lifespan,
)

# Add CORS middleware
if config.cors.enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allow_origins,
        allow_credentials=True,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
    )


# =============================================================================
# DEPENDENCY FUNCTIONS
# =============================================================================


def get_adapter() -> LlamaCppAdapter:
    """
    Dependency function to get the adapter instance.

    Returns:
        The global adapter instance

    Raises:
        ModelNotLoadedError: If adapter is not initialized
    """
    global adapter
    if not adapter:
        raise ModelNotLoadedError("Model adapter not initialized")
    return adapter


def get_adapter_optional() -> LlamaCppAdapter | None:
    """
    Dependency function to get the adapter instance or None if not available.

    Returns:
        The global adapter instance or None
    """
    global adapter
    return adapter


@app.exception_handler(ArtisanEngineError)
async def artisan_exception_handler(request: Request, exc: ArtisanEngineError):
    """Handle custom artisan engine exceptions."""
    return JSONResponse(
        status_code=400 if isinstance(exc, ValidationError) else 500,
        content=ErrorResponse(
            error=exc.message,
            error_code=exc.error_code,
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=f"Internal server error: {str(exc)}",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump(),
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check(
    # THE FIX: Define include_details as a direct query parameter
    include_details: bool = Query(
        False, description="Include detailed health information"
    ),
    adapter: LlamaCppAdapter | None = Depends(get_adapter_optional),
):
    """
    Health check endpoint.
    Returns the current health status of the API and model.
    """
    status = "healthy"
    model_status = None

    if adapter:
        try:
            model_status = adapter.health_check()
            if (
                not model_status.get("model_loaded", False)
                and not config.model.lazy_loading
            ):
                status = "degraded"
        except Exception:
            status = "unhealthy"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow().isoformat(),
        version=config.version,
        # This logic will now work correctly
        model_status=model_status if include_details else None,
        uptime=time.time() - start_time,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models(adapter: LlamaCppAdapter | None = Depends(get_adapter_optional)):
    """
    List available models (OpenAI-compatible endpoint).

    Returns information about available models.
    """
    models = []

    if adapter:
        model_info = adapter.get_model_info()
        serializable_params = {
            key: str(value) for key, value in model_info.items()
        }
        models.append(
            ModelInfo(
                id="local-llm",
                name="Local Language Model",
                path=str(model_info.get("model_path")), # Ensure path is a string
                loaded=model_info.get("is_loaded", False),
                parameters=serializable_params, # Use the sanitized dictionary
            )
        )

    return ModelsResponse(data=models)


@app.get("/schemas")
async def list_available_schemas():
    """
    List available schemas and cache information.

    Shows both pre-registered schemas and dynamic schema cache statistics.
    """
    cache_info = get_schema_cache_info()

    return {
        "registered_schemas": cache_info["registered_schema_names"],
        "registered_count": cache_info["registered_schemas"],
        "dynamic_schemas_cached": cache_info["dynamic_schemas_cached"],
        "cache_keys": cache_info["cache_keys"],
        "total_schemas_available": cache_info["registered_schemas"]
        + cache_info["dynamic_schemas_cached"],
        "note": "This API can handle any JSON Schema via dynamic creation, not just the registered ones listed above.",
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_structured(
    request: GenerateRequest, adapter: LlamaCppAdapter = Depends(get_adapter)
):
    """
    Generate structured JSON output using a specified schema.

    This is the main endpoint for structured generation.
    """

    # Get schema class
    schema_class = get_schema(request.schema_name)
    if not schema_class:
        raise ValidationError(f"Unknown schema: {request.schema_name}")

    try:
        # Generate structured output
        start_gen_time = time.time()

        result_obj = adapter.generate_structured(
            prompt=request.prompt,
            schema=schema_class,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **(request.extra_params or {}),
        )

        generation_time = time.time() - start_gen_time

        # Convert to JSON string and dict for response
        result_json = result_obj.model_dump_json()
        result_dict = result_obj.model_dump()

        return GenerateResponse(
            success=True,
            result=result_json,
            parsed_result=result_dict,
            schema_name=request.schema_name,
            generation_time=generation_time,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )

    except Exception as e:
        raise GenerationError(f"Generation failed: {e}")


@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def chat_completions(
    request: OpenAIChatRequest, adapter: LlamaCppAdapter = Depends(get_adapter)
):
    """
    OpenAI-compatible chat completions endpoint.

    Provides full OpenAI API compatibility with structured output support.
    Supports the official response_format structure used by the OpenAI client library.
    """

    # Extract prompt from messages
    if not request.messages:
        raise ValidationError("No messages provided")

    # Combine messages into a single prompt
    prompt_parts = []
    for message in request.messages:
        if message.role == "system":
            prompt_parts.append(f"System: {message.content}")
        elif message.role == "user":
            prompt_parts.append(f"User: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}")

    prompt = "\\n".join(prompt_parts)

    # Handle structured generation based on response_format
    if request.response_format:
        if request.response_format.type == "json_object":
            # For json_object type, we expect a json_schema
            if not request.response_format.json_schema:
                raise ValidationError(
                    "json_schema is required when response_format.type is 'json_object'"
                )

            # Find existing schema or dynamically create one
            json_schema_dict = request.response_format.json_schema.model_dump()
            schema_class = find_or_create_schema(json_schema_dict)

            # Generate structured output
            result_obj = adapter.generate_structured(
                prompt=prompt,
                schema=schema_class,
                max_tokens=request.max_tokens or 256,
                temperature=request.temperature or 1.0,
            )

            # Convert to JSON string for the response
            result = result_obj.model_dump_json()

        else:
            raise ValidationError(
                f"Unsupported response_format type: {request.response_format.type}"
            )
    else:
        # No structured output requested - this would be free-form generation
        # For now, we require structured output
        raise ValidationError(
            "This endpoint currently requires structured output. "
            "Please provide a response_format with type 'json_object'."
        )

    # Format as OpenAI response
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_timestamp = int(time.time())

    choice = OpenAIChoice(
        index=0,
        message=OpenAIMessage(role="assistant", content=result),
        finish_reason="stop",
    )

    return OpenAIChatResponse(
        id=response_id,
        created=created_timestamp,
        model=request.model,
        choices=[choice],
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(result.split()),
            "total_tokens": len(prompt.split()) + len(result.split()),
        },
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Artisan Engine",
        "version": config.version,
        "description": "Production-grade OpenAI-compatible API for local LLMs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "schemas": "/schemas",
            "generate": "/generate",
            "chat": "/v1/chat/completions",
        },
        "documentation": "/docs",
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for running the API server."""
    config = get_config()
    setup_logging(config)

    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        reload=config.server.reload,
        log_level=config.server.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
