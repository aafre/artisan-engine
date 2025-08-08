# Artisan Engine

**A production-grade, OpenAI-compatible API layer for local LLMs with guaranteed structured output.**

[![CI](https://github.com/aafre/artisan-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aafre/artisan-engine/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/artisan-engine.svg)](https://badge.fury.io/py/artisan-engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Mission

The goal of Artisan Engine is to bridge the last-mile gap between powerful open-source models and the developers who want to use them. It provides the elegant developer experience of a cloud API with the security and control of local infrastructure, making it simple to build production-grade AI applications on your own terms.

---

### Why Artisan?

Running local LLMs is powerful, but it's often a frustrating experience. Getting models to consistently produce valid JSON is a brittle process of prompt engineering and error handling. Artisan solves this problem.

* **100% Reliability:** Don't just *prompt* for JSON, *enforce* it. Artisan uses grammar-based sampling to guarantee that the model's output will always be a syntactically correct JSON object that validates against your Pydantic schema.
* **OpenAI Compatibility:** Use the official `openai` client library you already know. Just change the `base_url`, and your existing code works. This makes Artisan instantly compatible with LangChain, LlamaIndex, and the entire AI ecosystem.
* **One-Command Deploy:** Run the entire engine with a single `docker run` command using our pre-built image.
* **Language Agnostic:** Any service that can make an HTTP request (NodeJS, Go, Rust, Java, etc.) can use Artisan's power.

---

### Quick Start (Using Docker)

Get up and running in minutes with the pre-built Docker image from the GitHub Container Registry.

**Prerequisites:**
* Docker installed.
* A GGUF-format model file downloaded to your machine.

**1. Prepare Your Model Directory**

Create a directory on your computer and place your downloaded `.gguf` model file inside it.

```bash
# For example:
mkdir my-local-model
mv ~/Downloads/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf ./my-local-model/
```

**2. Run the Artisan Engine Container**

Run the following command from the directory containing your `my-local-model` folder. This command will download and start the latest version of Artisan Engine.

* **For Linux/macOS:**
    ```bash
    docker run -d --rm \\
      -p 8000:8000 \\
      -v "$(pwd)/my-local-model:/app/models" \\
      -e ARTISAN_MODEL_PATH="/app/models/your_model_name.gguf" \\
      --name artisan-engine \\
      ghcr.io/aafre/artisan-engine:latest
    ```

* **For Windows (PowerShell):**
    ```powershell
    docker run -d --rm `
      -p 8000:8000 `
      -v ${PWD}/my-local-model:/app/models `
      -e ARTISAN_MODEL_PATH="/app/models/your_model_name.gguf" `
      --name artisan-engine `
      ghcr.io/aafre/artisan-engine:latest
    ```

> **Note:** Replace `your_model_name.gguf` with the actual filename of your model and `aafre/artisan-engine` with your actual GitHub repository name.

The server will now be running at `http://localhost:8000`.

**3. Test with Python (OpenAI Client)**

This is the best way to verify that everything is working.

First, install the `openai` library: `pip install openai pydantic`

Then, run this Python script:
```python
import openai
from pydantic import BaseModel, Field

# 1. Define your desired Pydantic schema
class UserProfile(BaseModel):
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")

# 2. Point the OpenAI client to your local Artisan server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 3. Make the API call with the schema
response = client.chat.completions.create(
    model="local-llm",
    messages=[
        {"role": "user", "content": "Extract data for John Doe, who is 42 years old."}
    ],
    response_format={
        "type": "json_object",
        "json_schema": UserProfile.model_json_schema()
    }
)

# 4. The result is a guaranteed valid JSON string
json_response = response.choices[0].message.content
print("Raw JSON from server:", json_response)

# 5. You can load it directly into your Pydantic model
user = UserProfile.model_validate_json(json_response)
print(f"\\nSuccessfully validated object: {user}")
```

---

### Usage Examples

The `examples/` directory in this repository contains more runnable scripts that demonstrate how to use the Artisan Engine for common tasks, such as sentiment analysis and data extraction.

---

### Configuration

Artisan Engine is configured via environment variables. The most important variable is `ARTISAN_MODEL_PATH`, which is required for the server to start.

For a full list of configuration options (including CORS, GPU layers, and server settings), please see the `.env.example` file.

---

### Endpoints

* `/docs`: Interactive API documentation (Swagger UI).
* `/health`: Health check for the service and model.
* `/models`: Lists the available models (OpenAI-compatible).
* `/v1/chat/completions`: The OpenAI-compatible endpoint for structured and unstructured chat.

---

### Roadmap

Artisan Engine v0.1.0 is focused on providing guaranteed, schema-enforced output. Our future roadmap includes:

* **Full Function Calling / Tool Use:** Complete orchestration for multi-step agentic workflows.
* **The Assistants API:** A stateful, persistent API for managing long-running conversations with memory.
* **Integrated RAG:** Seamlessly connect your private documents to your local models.
* **Expanded Backend Support:** Official adapters for Ollama, vLLM, and other popular model servers.

---

### Contributing

Contributions are welcome! Please open an issue or submit a pull request.