import openai
from pydantic import BaseModel, Field

# 1. Define the schema you want the model to return
class UserProfile(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    is_premium_member: bool

# 2. Point the official OpenAI client to your local Artisan server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed" # The API key can be any string
)

print("--- Sending request to local Artisan Engine ---")

# 3. Make the API call, just like you would with OpenAI
response = client.chat.completions.create(
    model="local-llm", # This can be any string, as our server uses its configured model
    messages=[
        {"role": "user", "content": "Extract the profile for Sarah Connor. She is 35 and is a premium member."}
    ],
    # The response_format object is the key to structured output
    response_format={
        "type": "json_object",
        "json_schema": UserProfile.model_json_schema()
    }
)

# 4. The result is a guaranteed valid JSON string
json_result = response.choices[0].message.content
print("\nRaw JSON response from server:")
print(json_result)

# 5. You can load it directly into your Pydantic model without errors
user = UserProfile.model_validate_json(json_result)
print(f"\nSuccessfully validated Pydantic object:")
print(user)