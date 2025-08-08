# Artisan Engine

**A production-grade, OpenAI-compatible API layer for local LLMs with guaranteed structured output.**

[![CI](https://github.com/your-username/artisan-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/artisan-engine/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/artisan-engine.svg)](https://badge.fury.io/py/artisan-engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Mission

The goal of Artisan Engine is to bridge the last-mile gap between powerful open-source models and the developers who want to use them. It provides the elegant developer experience of a cloud API with the security and control of local infrastructure, making it simple to build production-grade AI applications on your own terms.

Artisan Engine provides a robust, self-hosted API that empowers you to run powerful open-source language models on your own infrastructure. It acts as a drop-in replacement for the OpenAI API, allowing you to switch from cloud to local models with a one-line code change.

Its core feature is **guaranteed, schema-enforced JSON output**, turning probabilistic models into deterministic API endpoints you can rely on for production applications.

---

### Why Artisan?

Running local LLMs is powerful, but it's often a frustrating experience. Getting models to consistently produce valid JSON is a brittle process of prompt engineering and error handling. Artisan solves this problem.

* **100% Reliability:** Don't just *prompt* for JSON, *enforce* it. Artisan uses grammar-based sampling to guarantee that the model's output will always be a syntactically correct JSON object that validates against your Pydantic schema.
* **OpenAI Compatibility:** Use the official `openai` client library you already know. Just change the `base_url`, and your existing code works. This makes Artisan instantly compatible with LangChain, LlamaIndex, and the entire AI ecosystem.
* **One-Command Deploy:** Get up and running in minutes with our optimized Docker setup. No complex dependency management or manual model downloads.
* **Language Agnostic:** Any service that can make an HTTP request (NodeJS, Go, Rust, Java, etc.) can use Artisan's power.

---

### Quick Start (Docker)

The easiest way to get started is with Docker. This will start the Artisan API server and a pre-configured `llama.cpp` server with a model.

**Prerequisites:**
* Docker and Docker Compose installed.
* Git installed.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/artisan-engine.git](https://github.com/your-username/artisan-engine.git)
cd artisan-engine
```

**2. Start the services:**
This command will download the necessary Docker images, download a default LLM model (`Llama-3.1-8B-Instruct`), and start both the Artisan API and the model server.
```bash
docker-compose up -d
```
The server will be available at `http://localhost:8000`.

**3. Test with cURL:**
Let's ask the server to extract data using the pre-registered `Invoice` schema.
```bash
curl -X POST http://localhost:8000/generate \\
-H "Content-Type: application/json" \\
-d '{
  "prompt": "Create an invoice for ACME Corp, ID INV-001. They bought 2 Super Widgets at 99.99 and 10 Standard Bolts at 0.50. Tax was 20.50.",
  "schema_name": "Invoice"
}'
```

**4. Test with Python (OpenAI Client):**
This demonstrates the power of the OpenAI-compatible endpoint.

First, install the `openai` library: `pip install openai`

Then, run this Python script:
```python
import openai
from pydantic import BaseModel, Field
from typing import List

# 1. Define your desired Pydantic schema
class User(BaseModel):
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
        "schema": User.model_json_schema()
    }
)

# 4. The result is a guaranteed valid JSON string
json_response = response.choices[0].message.content
print(json_response)

# You can load it directly into your Pydantic model without errors
user = User.model_validate_json(json_response)
print(f"\\nSuccessfully extracted user: {user.name}, Age: {user.age}")
```

---

### Usage Examples

The `examples/` directory contains runnable scripts that demonstrate how to use the Artisan Engine for common tasks. We encourage you to run them to see the engine in action after you have the server running.

* **[`examples/01_simple_sentiment_analysis.py`](./examples/01_simple_sentiment_analysis.py):** Shows how to use the powerful, OpenAI-compatible endpoint for a common NLP task with a schema that is defined entirely on the client-side.

* **[`examples/native_api_usage.py`](./examples/native_api_usage.py):** Demonstrates how to use the simple, native `/generate` endpoint with a schema that is pre-registered on the server.

To run an example, use the following command from your project root:
```bash
python examples/01_simple_sentiment_analysis.py
```

---

### Configuration

Artisan Engine supports flexible configuration through `.env` files and environment variables.

#### Quick Setup with .env File

1. **Copy the example configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your settings:**
   ```bash
   # Essential settings
   ARTISAN_MODEL_PATH=./models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
   ARTISAN_SERVER_PORT=8000
   ARTISAN_DEBUG=true
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

#### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTISAN_MODEL_PATH` | - | Path to your GGUF model file |
| `ARTISAN_SERVER_HOST` | `127.0.0.1` | Server host address |
| `ARTISAN_SERVER_PORT` | `8000` | Server port |
| `ARTISAN_MODEL_N_CTX` | `2048` | Context window size |
| `ARTISAN_MODEL_N_GPU_LAYERS` | `0` | GPU layers (0=CPU, -1=all GPU) |
| `ARTISAN_DEBUG` | `false` | Enable debug mode |

#### Example Configurations

**Local Development:**
```bash
ARTISAN_MODEL_PATH=./models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
ARTISAN_DEBUG=true
ARTISAN_SERVER_RELOAD=true
ARTISAN_SERVER_LOG_LEVEL=debug
```

**Production Deployment:**
```bash
ARTISAN_ENVIRONMENT=production
ARTISAN_MODEL_PATH=/opt/models/production-model.gguf
ARTISAN_SERVER_HOST=0.0.0.0
ARTISAN_SERVER_PORT=8080
ARTISAN_DEBUG=false
ARTISAN_CORS_ALLOW_ORIGINS=https://yourdomain.com
```

**GPU Acceleration:**
```bash
ARTISAN_MODEL_N_GPU_LAYERS=32
ARTISAN_MODEL_N_CTX=4096
```

> **Note:** Environment variables take precedence over `.env` file values. See `.env.example` for all available options.

---

### Endpoints

* `/docs`: Interactive API documentation (Swagger UI).
* `/health`: Health check for the service and model.
* `/models`: Lists the available models (OpenAI-compatible).
* `/schemas`: Lists the pre-registered schemas available for the `/generate` endpoint.
* `/generate`: Artisan's native endpoint for structured generation using pre-registered schemas.
* `/v1/chat/completions`: The OpenAI-compatible endpoint that accepts dynamic schemas.

---

### Roadmap

Artisan Engine v0.1.0 is focused on providing guaranteed, schema-enforced output. Our future roadmap includes:

* **Full Function Calling / Tool Use:** Complete orchestration for multi-step agentic workflows.
* **The Assistants API:** A stateful, persistent API for managing long-running conversations with memory, compatible with OpenAI's `Threads` and `Runs`.
* **Integrated RAG:** Seamlessly connect your private documents to your local models.
* **Expanded Backend Support:** Official adapters for Ollama, vLLM, and other popular model servers.

---

### Contributing

Contributions are welcome! Please open an issue or submit a pull request.