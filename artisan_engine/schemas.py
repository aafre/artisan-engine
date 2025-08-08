"""
Schema definitions and registry for structured generation.

This module contains example schemas and the schema registry system,
separate from the core API models.
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field, create_model

# =============================================================================
# EXAMPLE SCHEMAS
# =============================================================================


class User(BaseModel):
    """Example schema for user data extraction."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    is_subscribed: bool = Field(description="Whether the user is a subscriber")
    interests: list[str] = Field(description="A list of the user's interests")


class LineItem(BaseModel):
    """Line item for invoices."""

    item: str
    quantity: int
    price: float


class Invoice(BaseModel):
    """Example schema for invoice data extraction."""

    invoice_id: str
    customer_name: str
    items: list[LineItem]
    tax_amount: float | None = None


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

# Registry of available schemas for dynamic lookup
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "User": User,
    "Invoice": Invoice,
    "LineItem": LineItem,
}


def get_schema(schema_name: str) -> type[BaseModel] | None:
    """
    Get a schema class by name.

    Args:
        schema_name: Name of the schema to retrieve

    Returns:
        Schema class if found, None otherwise
    """
    return SCHEMA_REGISTRY.get(schema_name)


def list_schemas() -> list[str]:
    """
    Get list of available schema names.

    Returns:
        List of available schema names
    """
    return list(SCHEMA_REGISTRY.keys())


def register_schema(name: str, schema_class: type[BaseModel]) -> None:
    """
    Register a new schema in the registry.

    Args:
        name: Name to register the schema under
        schema_class: Pydantic model class
    """
    SCHEMA_REGISTRY[name] = schema_class


def schema_to_json_schema(schema_class: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic schema to JSON Schema format for OpenAI compatibility.

    Args:
        schema_class: Pydantic model class

    Returns:
        JSON Schema dictionary
    """
    return schema_class.model_json_schema()


# =============================================================================
# DYNAMIC SCHEMA CREATION
# =============================================================================

# Cache for dynamically created schemas to avoid recreating identical ones
_DYNAMIC_SCHEMA_CACHE: dict[str, type[BaseModel]] = {}


def _json_schema_to_pydantic_type(property_schema: dict[str, Any]) -> Any:
    """
    Convert a JSON Schema property definition to a Pydantic-compatible type.

    Args:
        property_schema: JSON Schema property definition

    Returns:
        Pydantic-compatible type annotation
    """
    json_type = property_schema.get("type", "string")

    if json_type == "string":
        return str
    elif json_type == "integer":
        return int
    elif json_type == "number":
        return float
    elif json_type == "boolean":
        return bool
    elif json_type == "array":
        # Handle array types
        items_schema = property_schema.get("items", {"type": "string"})
        item_type = _json_schema_to_pydantic_type(items_schema)
        return list[item_type]
    elif json_type == "object":
        # For nested objects, we'd need recursive schema creation
        # For now, return Dict[str, Any] as a fallback
        return dict[str, Any]
    else:
        # Default to Any for unknown types
        return Any


def create_schema_from_json_schema(json_schema: dict[str, Any]) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model from a JSON Schema definition.

    This is the core function that makes our API truly universal - it can handle
    any JSON Schema sent by the OpenAI client, not just pre-registered ones.

    Args:
        json_schema: JSON Schema dictionary

    Returns:
        Dynamically created Pydantic model class

    Raises:
        ValueError: If the JSON Schema is invalid or unsupported
    """
    if json_schema.get("type") != "object":
        raise ValueError("Only object-type JSON schemas are supported")

    properties = json_schema.get("properties", {})
    required_fields = set(json_schema.get("required", []))

    if not properties:
        raise ValueError("JSON Schema must have at least one property")

    # Create a unique cache key based on the schema content
    schema_key = hashlib.md5(
        json.dumps(json_schema, sort_keys=True).encode()
    ).hexdigest()

    # Return cached schema if it exists
    if schema_key in _DYNAMIC_SCHEMA_CACHE:
        return _DYNAMIC_SCHEMA_CACHE[schema_key]

    # Build field definitions for Pydantic create_model
    field_definitions = {}

    for field_name, property_schema in properties.items():
        # Get the Python type for this field
        python_type = _json_schema_to_pydantic_type(property_schema)

        # Handle optional vs required fields
        if field_name not in required_fields:
            python_type = python_type | None
            default_value = None
        else:
            default_value = ...  # Required field marker

        # Create Field with description if available
        description = property_schema.get("description")
        if description:
            field_definitions[field_name] = (
                python_type,
                Field(default=default_value, description=description),
            )
        else:
            field_definitions[field_name] = (python_type, default_value)

    # Create dynamic model class
    model_name = f"DynamicSchema_{schema_key[:8]}"

    try:
        dynamic_model = create_model(model_name, **field_definitions)

        # Cache the created model
        _DYNAMIC_SCHEMA_CACHE[schema_key] = dynamic_model

        return dynamic_model

    except Exception as e:
        raise ValueError(
            f"Failed to create Pydantic model from JSON Schema: {e}"
        ) from e


def find_or_create_schema(json_schema: dict[str, Any]) -> type[BaseModel]:
    """
    Find a matching registered schema OR dynamically create one from JSON Schema.

    This function first tries to find a matching pre-registered schema for performance,
    but falls back to dynamic creation for maximum flexibility.

    Args:
        json_schema: JSON Schema dictionary

    Returns:
        Matching or newly created Pydantic model class
    """
    # First, try to find an exact match in registered schemas (for performance)
    if "properties" in json_schema and "required" in json_schema:
        target_props = set(json_schema["properties"].keys())
        target_required = set(json_schema.get("required", []))

        for schema_class in SCHEMA_REGISTRY.values():
            schema_json = schema_class.model_json_schema()

            if (
                "properties" in schema_json
                and set(schema_json["properties"].keys()) == target_props
                and set(schema_json.get("required", [])) == target_required
            ):
                # Do a deeper check on property types to avoid the vulnerability
                type_match = True
                for prop_name, prop_def in json_schema["properties"].items():
                    if prop_name in schema_json["properties"]:
                        schema_prop = schema_json["properties"][prop_name]
                        if prop_def.get("type") != schema_prop.get(
                            "type"
                        ) or prop_def.get("items", {}).get("type") != schema_prop.get(
                            "items", {}
                        ).get("type"):
                            type_match = False
                            break

                if type_match:
                    return schema_class

    # No exact match found, create dynamic schema
    return create_schema_from_json_schema(json_schema)


def get_schema_cache_info() -> dict[str, Any]:
    """
    Get information about the dynamic schema cache.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "registered_schemas": len(SCHEMA_REGISTRY),
        "dynamic_schemas_cached": len(_DYNAMIC_SCHEMA_CACHE),
        "registered_schema_names": list(SCHEMA_REGISTRY.keys()),
        "cache_keys": list(_DYNAMIC_SCHEMA_CACHE.keys()),
    }
