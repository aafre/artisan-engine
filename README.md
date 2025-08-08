# Artisan Engine

**A production-grade, OpenAI-compatible API layer for local LLMs with guaranteed structured output.**

[![CI](https://github.com/aafre/artisan-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aafre/artisan-engine/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/artisan-engine.svg)](https://badge.fury.io/py/artisan-engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Mission

The goal of Artisan Engine is to bridge the last-mile gap between powerful open-source models and the developers who want to use them. It provides the elegant developer experience of a cloud API with the security and control of local infrastructure, making it simple to build production-grade AI applications on your own terms.

---

### Project Status & Roadmap

Artisan Engine is currently in its initial `v0.1.0` release. The core focus of this version is to deliver a rock-solid, OpenAI-compatible endpoint for **guaranteed structured output**.

Our future roadmap is focused on building a complete, stateful application platform:

* [ ] **Full Function Calling / Tool Use:** Complete orchestration for multi-step agentic workflows.
* [ ] **The Assistants API:** A stateful, persistent API for managing long-running conversations with memory.
* [ ] **Integrated RAG:** Seamlessly connect your private documents to your local models.
* [ ] **Expanded Backend Support:** Official adapters for Ollama, vLLM, and other popular model servers.

We are actively looking for contributors to help us build this future. See the "Contributing" section below!

---

### Key Features

* **Guaranteed Structured Output:** Don't just *prompt* for JSON, *enforce* it. Artisan uses grammar-based sampling to guarantee that the model's output will always be a syntactically correct JSON object that validates against your Pydantic schema.
* **OpenAI Compatibility:** Use the official `openai` client library you already know. Just change the `base_url`, and your existing code works.
* **One-Command Deploy:** A single `docker-compose up` command downloads the model and starts the server.
* **Language Agnostic:** Any service that can make an HTTP request (NodeJS, Go, Rust, Java, etc.) can use Artisan's power.

---

### Quick Start (with Docker Compose)

Get the entire engine running with a single command. This is the easiest and recommended way to get started.

**Prerequisites:**
* Docker and Docker Compose installed.
* Git installed.

**1. Clone the repository:**
```bash
git clone [https://github.com/aafre/artisan-engine.git](https://github.com/aafre/artisan-engine.git)
cd artisan-engine
```

**2. Start the services:**
This single command will take care of everything:
* Build the Artisan Engine image.
* **Automatically download a default LLM model** (`Llama-3.1-8B-Instruct`) if you don't have it.
* Start the Artisan API server.

```bash
docker-compose up -d
```
> **Note:** The first time you run this, it may take several minutes to download the multi-gigabyte model file. On subsequent runs, it will start instantly as the model is cached in a Docker volume.

The server will be available at `http://localhost:8000`.

**3. Test with Python (OpenAI Client)**

Once the server is running, you can verify everything is working with this script.

First, install the `openai` library: `pip install openai pydantic`

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

The `examples/` directory in this repository contains more runnable scripts that demonstrate how to use the Artisan Engine for common tasks.

---

### Configuration

Artisan Engine is configured via environment variables. The easiest way to configure the `docker-compose` setup is to edit the `environment` section for the `artisan-engine` service directly in the `docker-compose.yml` file.

For a full list of configuration options, please see the `.env.example` file.

---

### Endpoints

* `/docs`: Interactive API documentation (Swagger UI).
* `/health`: Health check for the service and model.
* `/models`: Lists the available models (OpenAI-compatible).
* `/v1/chat/completions`: The OpenAI-compatible endpoint for structured and unstructured chat.

---

### Powered By

Artisan Engine stands on the shoulders of giants. Our core functionality is made possible by these fantastic open-source projects:

* **[Outlines](https://github.com/dottxt-ai/outlines):** For the state-of-the-art, grammar-based generation that guarantees our structured output.
* **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python):** For high-performance inference of GGUF models on local hardware.
* **[FastAPI](https://fastapi.tiangolo.com/):** For building our robust and modern API.

---

### Contributing

Contributions are welcome and essential for making Artisan Engine the best tool for local AI development! We have several issues flagged as `good first issue` that are perfect for getting started. Please see the [issues tab](https://github.com/aafre/artisan-engine/issues) to get involved.