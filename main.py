# poc.py
#
# Objective: Prove that we can force a local LLM to generate a JSON object
# that strictly validates against a Pydantic schema.

from typing import List, Optional
import outlines
from llama_cpp import Llama
from pydantic import BaseModel, Field
import time
import os

# ==============================================================================
# 1. DEFINE THE INPUT SCHEMA
# ==============================================================================
class User(BaseModel):
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    is_subscribed: bool = Field(description="Whether the user is a subscriber")
    interests: list[str] = Field(description="A list of the user's interests")

class LineItem(BaseModel):
    item: str
    quantity: int
    price: float

class Invoice(BaseModel):
    invoice_id: str
    customer_name: str
    items: List[LineItem]
    tax_amount: Optional[float] = None # Test an optional field

# ==============================================================================
# 2. DEFINE THE INPUT TEXT
# ==============================================================================
# input_text = "Extract the details for John Smith. He is a 32-year-old active subscriber who enjoys machine learning, hiking, and cooking."
input_text = "Please create an invoice for customer 'ACME Corp'. The invoice ID is 'INV-001'. They bought 2 units of 'Super Widget' at 99.99 each and 10 units of 'Standard Bolt' at 0.50 each. The total tax was 20.50."


# ==============================================================================
# 3. LOAD THE MODEL
# ==============================================================================
model_path = "./Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
# model_path = "./Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found at {model_path}. "
        "Please follow Step 2 in the README to download it."
    )

print(f"Loading model from: {model_path} ... (This may take a few moments)")
# Your excellent model loading method
model = outlines.from_llamacpp(Llama(model_path))
print("Model loaded successfully.")

# ==============================================================================
# 4. THE CORE LOGIC: Constrained Generation
# ==============================================================================
# Your clean generator instantiation
generator = outlines.Generator(model, Invoice)

# ==============================================================================
# 5. EXECUTE THE GENERATOR
# ==============================================================================
print(f"\nPrompting model with: '{input_text}'")
print("Generating guaranteed Pydantic object...")
start_time = time.time()

# The result will be a complete JSON string.
json_string_result = generator(input_text, max_tokens=200)

end_time = time.time()

# ==============================================================================
# 6. VERIFY THE OUTCOME
# ==============================================================================
print(f"\n--- POC ATTEMPT (Completed in {end_time - start_time:.2f} seconds) ---")
print("\n1. Raw JSON string from model:")
print(json_string_result)

# Add robust parsing to ensure the final object is correct
try:
    invoice_object = Invoice.model_validate_json(json_string_result) # <--- THE FIX IS HERE
    print("\n--- POC SUCCESS ---")
    print("\n2. Validated Pydantic Object:")
    print(invoice_object)
    print("\n3. Object Type:")
    print(type(invoice_object))

    # Assertions to programmatically confirm correctness
    assert invoice_object.customer_name == "ACME Corp"
    assert len(invoice_object.items) == 2
    assert invoice_object.items[0].item == "Super Widget"
    print("\nAssertions passed! Data is semantically correct.")

except Exception as e:
    print(f"\n--- POC FAILED ---")
    print(f"Could not validate the JSON string. Error: {e}")
