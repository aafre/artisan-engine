import openai
from pydantic import BaseModel, Field


# 1. Define your desired Pydantic schema, completely on the client-side.
# The server does not need to know about this schema in advance.
class Product(BaseModel):
    product_id: str = Field(
        description="A unique identifier for the product, like a SKU."
    )
    name: str = Field(description="The name of the product.")
    tags: list[str] = Field(description="A list of relevant keywords or tags.")
    is_available: bool = Field(description="Whether the product is currently in stock.")


def extract_product_info():
    """
    Demonstrates using the OpenAI-compatible /v1/chat/completions endpoint
    with a dynamically defined schema.
    """
    print("\n--- 2. Testing OpenAI-Compatible /v1/chat/completions Endpoint ---")

    # 2. Point the official OpenAI client to your local Artisan server
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # Can be any string
    )

    prompt = (
        "Extract product information from the following description: "
        "'The new T-800 series AI assistant (SKU: AIBOT-T800) is a powerful "
        "tool for developers. Keywords: AI, assistant, robotics. Currently out of stock.'"
    )

    try:
        # 3. Make the API call, passing the schema definition directly
        response = client.chat.completions.create(
            model="local-llm",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "json_schema": Product.model_json_schema(),
            },
        )

        # 4. The result is a guaranteed valid JSON string
        json_response = response.choices[0].message.content
        print("\nRaw JSON response from server:")
        print(json_response)

        # 5. You can load it directly into your Pydantic model without errors
        product = Product.model_validate_json(json_response)
        print("\nSuccessfully validated Pydantic object:")
        print(product)

    except openai.APIError as e:
        print(f"An error occurred while calling the API: {e}")


if __name__ == "__main__":
    extract_product_info()
