"""LLM Adapter for constrained generation using outlines and llama.cpp."""

import logging
import time
from pathlib import Path
from typing import Any

import outlines
from llama_cpp import Llama
from pydantic import BaseModel

from .exceptions import GenerationError, ModelNotFoundError, ModelNotLoadedError

logger = logging.getLogger(__name__)


class LlamaCppAdapter:
    """
    Adapter class for llama.cpp models with structured generation capabilities.

    This class handles model loading, caching, and constrained generation using
    the outlines library to ensure output conforms to Pydantic schemas.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        lazy_loading: bool = True,
        n_ctx: int = 2048,
        **llama_kwargs: Any,
    ):
        """
        Initialize the LlamaCpp adapter.

        Args:
            model_path: Path to the GGUF model file
            lazy_loading: If True, delay model loading until first generation
            n_ctx: Context window size (number of tokens)
            **llama_kwargs: Additional arguments passed to Llama constructor
        """
        self.model_path = Path(model_path) if model_path else None
        self.lazy_loading = lazy_loading
        self.n_ctx = n_ctx

        # Merge n_ctx into llama_kwargs, but allow override if explicitly provided
        self.llama_kwargs = {"n_ctx": n_ctx, **llama_kwargs}

        # Model state
        self._llama_model: Llama | None = None
        self._outlines_model: Any | None = None
        self._generators_cache: dict[str, Any] = {}
        self._is_loaded = False

        # Load immediately if not lazy loading
        if not lazy_loading and model_path:
            self.load_model()

    def load_model(self, model_path: str | Path | None = None) -> None:
        """
        Load the GGUF model file.

        Args:
            model_path: Optional path to model file (overrides instance path)

        Raises:
            ModelNotFoundError: If model file doesn't exist
            GenerationError: If model loading fails
        """
        if model_path:
            self.model_path = Path(model_path)

        if not self.model_path:
            raise ModelNotFoundError("No model path provided")

        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {self.model_path}")

        try:
            logger.info(f"Loading model from: {self.model_path}")
            start_time = time.time()

            # Initialize llama.cpp model
            self._llama_model = Llama(str(self.model_path), **self.llama_kwargs)

            # Wrap with outlines
            self._outlines_model = outlines.from_llamacpp(self._llama_model)

            load_time = time.time() - start_time
            self._is_loaded = True

            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise GenerationError(f"Model loading failed: {e}") from e

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded and self._outlines_model is not None

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._llama_model:
            # llama.cpp doesn't have explicit cleanup, but we can clear references
            self._llama_model = None

        self._outlines_model = None
        self._generators_cache.clear()
        self._is_loaded = False
        logger.info("Model unloaded successfully")

    def _filter_generation_params(self, **kwargs: Any) -> dict[str, Any]:
        """
        Filter generation parameters to only include those supported by llama-cpp-python.
        
        Args:
            **kwargs: Raw generation parameters
            
        Returns:
            Dictionary with only valid parameters
        """
        # Common llama-cpp-python generation parameters
        valid_params = {
            'top_k', 'top_p', 'min_p', 'typical_p', 'temp', 'repeat_penalty',
            'repeat_last_n', 'frequency_penalty', 'presence_penalty', 'tfs_z',
            'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'stop',
            'seed', 'logprobs', 'echo', 'stopping_criteria', 'logit_bias',
            'grammar', 'max_tokens', 'temperature'  # Include the main ones too
        }
        
        filtered = {}
        ignored = []
        
        for key, value in kwargs.items():
            if key in valid_params:
                filtered[key] = value
            else:
                ignored.append(key)
        
        if ignored:
            logger.debug(f"Ignoring unsupported generation parameters: {ignored}")
            
        return filtered

    def _ensure_pydantic_result(self, result: Any, schema: type[BaseModel]) -> BaseModel:
        """
        Ensure the generator result is a valid Pydantic model instance.
        
        Args:
            result: The result from the outlines generator
            schema: The expected Pydantic model class
            
        Returns:
            A validated Pydantic model instance
            
        Raises:
            GenerationError: If the result cannot be converted to the expected schema
        """
        try:
            # Case 1: Already a Pydantic model of the correct type
            if isinstance(result, schema):
                logger.debug(f"Generator returned correct Pydantic type: {type(result)}")
                return result
            
            # Case 2: It's a Pydantic model but different type - extract dict and convert
            elif isinstance(result, BaseModel):
                logger.debug(f"Converting Pydantic model {type(result)} to {schema}")
                return schema.model_validate(result.model_dump())
            
            # Case 3: JSON string - parse to Pydantic object
            elif isinstance(result, str):
                logger.debug(f"Converting JSON string to {schema}")
                try:
                    return schema.model_validate_json(result)
                except Exception as json_err:
                    # If JSON parsing fails, try to parse as plain text that might be JSON-like
                    logger.debug(f"JSON parsing failed: {json_err}, trying dict conversion")
                    import json
                    parsed_dict = json.loads(result)
                    return schema.model_validate(parsed_dict)
            
            # Case 4: Dictionary - convert to Pydantic object  
            elif isinstance(result, dict):
                logger.debug(f"Converting dictionary to {schema}")
                return schema.model_validate(result)
            
            # Case 5: Unknown type - provide detailed error
            else:
                raise GenerationError(
                    f"Generator returned unexpected type {type(result)} (value: {result!r}). "
                    f"Expected {schema} or compatible format (str, dict, BaseModel)."
                )
                
        except Exception as e:
            logger.error(f"Failed to convert generator result to {schema}: {e}")
            raise GenerationError(
                f"Could not convert generator output to {schema}: {e}. "
                f"Result type: {type(result)}, value: {result!r}"
            ) from e

    def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        max_tokens: int = 200,
        temperature: float = 0.7,
        **generation_kwargs: Any,
    ) -> BaseModel:
        """
        Generate structured output that validates against a Pydantic schema.

        Args:
            prompt: Input text prompt
            schema: Pydantic model class defining the output structure
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-1.0)
            **generation_kwargs: Additional generation parameters

        Returns:
            Validated Pydantic model instance

        Raises:
            ModelNotLoadedError: If model is not loaded
            GenerationError: If generation fails
        """
        # Lazy load if needed
        if not self.is_loaded():
            if self.lazy_loading and self.model_path:
                self.load_model()
            else:
                raise ModelNotLoadedError("Model not loaded. Call load_model() first.")

        try:
            # Get or create generator for this schema
            schema_name = schema.__name__
            if schema_name not in self._generators_cache:
                logger.debug(f"Creating generator for schema: {schema_name}")
                self._generators_cache[schema_name] = outlines.Generator(
                    self._outlines_model, schema
                )

            generator = self._generators_cache[schema_name]

            # Filter and prepare generation parameters
            filtered_kwargs = self._filter_generation_params(**generation_kwargs)
            
            # Generate structured output
            logger.debug(f"Generating with prompt: {prompt[:100]}...")
            start_time = time.time()

            result = generator(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **filtered_kwargs,
            )

            generation_time = time.time() - start_time
            logger.debug(f"Generation completed in {generation_time:.2f} seconds")

            # Handle different return types from outlines generator
            return self._ensure_pydantic_result(result, schema)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Structured generation failed: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on the adapter.

        Returns:
            Dictionary with health status information
        """
        return {
            "model_loaded": self.is_loaded(),
            "model_path": str(self.model_path) if self.model_path else None,
            "generators_cached": len(self._generators_cache),
            "lazy_loading": self.lazy_loading,
            "n_ctx": self.n_ctx,
        }

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": str(self.model_path) if self.model_path else None,
            "is_loaded": self.is_loaded(),
            "lazy_loading": self.lazy_loading,
            "n_ctx_configured": self.n_ctx,
        }

        if self._llama_model and self.is_loaded():
            # Add llama.cpp specific info if available
            try:
                info.update(
                    {
                        "n_ctx_actual": getattr(self._llama_model, "n_ctx", "unknown"),
                        "n_vocab": getattr(self._llama_model, "n_vocab", "unknown"),
                    }
                )
            except Exception:
                # Some attributes might not be available depending on llama.cpp version
                pass

        return info
