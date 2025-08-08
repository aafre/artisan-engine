"""
Example 01: Simple Sentiment Analysis

This example demonstrates how to use the OpenAI-compatible endpoint for a
common NLP task: sentiment analysis.

The Pydantic schema for the response is defined directly in this client script,
and the server handles it dynamically without needing any pre-registration.
"""

from typing import Literal

import openai
from pydantic import BaseModel, Field


# =============================================================================
# 1. DEFINE THE DESIRED SCHEMA (Client-Side)
# =============================================================================
class SentimentResponse(BaseModel):
    """Schema for sentiment analysis output."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment of the text."
    )
    confidence: float = Field(
        description="A confidence score for the sentiment, between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )


# =============================================================================
# 2. SETUP THE CLIENT
# =============================================================================
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# =============================================================================
# 3. DEFINE TEXTS TO ANALYZE
# =============================================================================
texts_to_analyze = [
    "I absolutely love this new framework! It's intuitive and powerful.",
    "I'm really disappointed with the latest update. It's buggy and slow.",
    "The documentation is clear and the examples are helpful.",
    "The server was down for maintenance yesterday.",
]

# =============================================================================
# 4. RUN THE ANALYSIS
# =============================================================================
for text in texts_to_analyze:
    print(f"--- Analyzing text: '{text}' ---")
    try:
        response = client.chat.completions.create(
            model="local-llm",
            messages=[
                {
                    "role": "user",
                    "content": f"Perform sentiment analysis on the following text. The sentiment must be one of 'positive', 'negative', or 'neutral' in all lowercase. Text: '{text}'",
                }
            ],
            response_format={
                "type": "json_object",
                "json_schema": SentimentResponse.model_json_schema(),
            },
        )

        json_result = response.choices[0].message.content
        sentiment = SentimentResponse.model_validate_json(json_result)

        print(
            f"  -> Sentiment: {sentiment.sentiment} (Confidence: {sentiment.confidence:.2f})"
        )

    except openai.APIError as e:
        print(f"Failed to analyze text. Error: {e}")

    print("-" * 20)
