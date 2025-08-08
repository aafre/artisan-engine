"""
Tests for dynamic schema creation and registry functionality.
"""

import pytest
from pydantic import BaseModel

from artisan_engine.schemas import (
    _DYNAMIC_SCHEMA_CACHE,
    Invoice,
    LineItem,
    User,
    create_schema_from_json_schema,
    find_or_create_schema,
    get_schema,
    get_schema_cache_info,
    list_schemas,
    register_schema,
)


class TestSchemaRegistry:
    """Test the schema registry functionality."""

    def test_get_registered_schema(self):
        """Test retrieving registered schemas."""
        user_schema = get_schema("User")
        assert user_schema == User

        invoice_schema = get_schema("Invoice")
        assert invoice_schema == Invoice

    def test_get_nonexistent_schema(self):
        """Test retrieving non-existent schema."""
        schema = get_schema("NonExistent")
        assert schema is None

    def test_list_schemas(self):
        """Test listing available schemas."""
        schemas = list_schemas()
        assert isinstance(schemas, list)
        assert "User" in schemas
        assert "Invoice" in schemas
        assert "LineItem" in schemas

    def test_register_new_schema(self):
        """Test registering a new schema."""

        class TestProduct(BaseModel):
            name: str
            price: float

        register_schema("TestProduct", TestProduct)

        # Should now be retrievable
        retrieved = get_schema("TestProduct")
        assert retrieved == TestProduct

        # Should appear in list
        assert "TestProduct" in list_schemas()


class TestDynamicSchemaCreation:
    """Test dynamic schema creation from JSON schemas."""

    def test_simple_schema_creation(self):
        """Test creating a simple schema."""
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
                "active": {"type": "boolean"},
            },
            "required": ["name", "age"],
        }

        schema_class = create_schema_from_json_schema(json_schema)

        # Test the created schema
        instance = schema_class(name="John", age=30, active=True)
        assert instance.name == "John"
        assert instance.age == 30
        assert instance.active is True

        # Test required field validation
        with pytest.raises(ValueError):
            schema_class(age=30)  # Missing required 'name'

    def test_optional_fields(self):
        """Test schema with optional fields."""
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["title"],
        }

        schema_class = create_schema_from_json_schema(json_schema)

        # Test with optional field
        instance = schema_class(title="Test", description="A test")
        assert instance.title == "Test"
        assert instance.description == "A test"

        # Test without optional field
        instance2 = schema_class(title="Test Only")
        assert instance2.title == "Test Only"
        assert instance2.description is None

    def test_array_fields(self):
        """Test schema with array fields."""
        json_schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "scores": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["tags"],
        }

        schema_class = create_schema_from_json_schema(json_schema)

        instance = schema_class(tags=["python", "ai"], scores=[95, 87])
        assert instance.tags == ["python", "ai"]
        assert instance.scores == [95, 87]

    def test_mixed_types(self):
        """Test schema with various field types."""
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"},
                "items": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            "required": ["name", "count"],
        }

        schema_class = create_schema_from_json_schema(json_schema)

        instance = schema_class(
            name="Product",
            count=10,
            price=29.99,
            active=True,
            items=["item1", "item2"],
            metadata={"key": "value"},
        )

        assert instance.name == "Product"
        assert instance.count == 10
        assert instance.price == 29.99
        assert instance.active is True
        assert instance.items == ["item1", "item2"]
        assert instance.metadata == {"key": "value"}

    def test_schema_caching(self):
        """Test that identical schemas are cached."""
        json_schema = {
            "type": "object",
            "properties": {"test": {"type": "string"}},
            "required": ["test"],
        }

        # Clear cache first
        _DYNAMIC_SCHEMA_CACHE.clear()

        schema1 = create_schema_from_json_schema(json_schema)
        schema2 = create_schema_from_json_schema(json_schema)

        # Should be the same class object (cached)
        assert schema1 is schema2
        assert len(_DYNAMIC_SCHEMA_CACHE) == 1

    def test_invalid_schema(self):
        """Test error handling for invalid schemas."""
        # Missing type
        with pytest.raises(ValueError):
            create_schema_from_json_schema({"properties": {"test": {"type": "string"}}})

        # Wrong type
        with pytest.raises(ValueError):
            create_schema_from_json_schema({"type": "array"})

        # No properties
        with pytest.raises(ValueError):
            create_schema_from_json_schema({"type": "object"})


class TestFindOrCreateSchema:
    """Test the find_or_create_schema function."""

    def test_finds_existing_schema(self):
        """Test that existing schemas are found correctly."""
        # Create JSON schema that matches User
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "is_subscribed": {"type": "boolean"},
                "interests": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age", "is_subscribed", "interests"],
        }

        schema_class = find_or_create_schema(json_schema)

        # Should return the registered User schema
        assert schema_class == User

    def test_creates_new_schema(self):
        """Test that new schemas are created when no match exists."""
        json_schema = {
            "type": "object",
            "properties": {
                "unique_field": {"type": "string"},
                "another_unique": {"type": "integer"},
            },
            "required": ["unique_field"],
        }

        schema_class = find_or_create_schema(json_schema)

        # Should not be any of our registered schemas
        assert schema_class != User
        assert schema_class != Invoice
        assert schema_class != LineItem

        # Should be able to create instances
        instance = schema_class(unique_field="test", another_unique=42)
        assert instance.unique_field == "test"
        assert instance.another_unique == 42

    def test_type_mismatch_creates_new(self):
        """Test that type mismatches create new schemas instead of matching."""
        # Schema with same field names as User but different types
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "integer"},  # Different type!
                "age": {"type": "string"},  # Different type!
                "is_subscribed": {"type": "boolean"},
                "interests": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age", "is_subscribed", "interests"],
        }

        schema_class = find_or_create_schema(json_schema)

        # Should NOT return User due to type mismatch
        assert schema_class != User

        # Should create a working schema with the different types
        instance = schema_class(
            name=123,  # integer, not string
            age="thirty",  # string, not int
            is_subscribed=True,
            interests=["test"],
        )
        assert instance.name == 123
        assert instance.age == "thirty"


class TestSchemaCacheInfo:
    """Test schema cache information."""

    def test_cache_info(self):
        """Test getting cache information."""
        # Clear cache
        _DYNAMIC_SCHEMA_CACHE.clear()

        info = get_schema_cache_info()

        assert "registered_schemas" in info
        assert "dynamic_schemas_cached" in info
        assert "registered_schema_names" in info
        assert "cache_keys" in info

        assert info["registered_schemas"] >= 3  # User, Invoice, LineItem
        assert "User" in info["registered_schema_names"]

        # Create a dynamic schema
        json_schema = {
            "type": "object",
            "properties": {"test": {"type": "string"}},
            "required": ["test"],
        }
        create_schema_from_json_schema(json_schema)

        # Check cache was updated
        updated_info = get_schema_cache_info()
        assert updated_info["dynamic_schemas_cached"] == 1
        assert len(updated_info["cache_keys"]) == 1
