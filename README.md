# Artisan Engine 🎨

**A production-grade, OpenAI-compatible API layer for local LLMs with guaranteed structured output.**

[![CI](https://github.com/aafre/artisan-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aafre/artisan-engine/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/artisan-engine.svg)](https://badge.fury.io/py/artisan-engine)
[![Docker Image](https://ghcr-badge.egpl.dev/aafre/artisan-engine/size)](https://github.com/aafre/artisan-engine/pkgs/container/artisan-engine)

## Mission

The goal of Artisan Engine is to bridge the last-mile gap between powerful open-source models and the developers who want to use them. It provides the elegant developer experience of a cloud API with the security and control of local infrastructure, making it simple to build production-grade AI applications on your own terms.

---

### Key Features

* **Guaranteed Structured Output:** Don't just *prompt* for JSON, *enforce* it. Artisan uses grammar-based sampling to guarantee that the model's output will always be a syntactically correct JSON object that validates against your Pydantic schema.
* **OpenAI Compatibility:** Use the official `openai` client library you already know. Just change the `base_url`, and your existing code works.
* **Simple Deployment:** Get up and running with a single `docker run` command or `pip install`.
* **Language Agnostic:** Any service that can make an HTTP request can use Artisan's power.

---

### Installation & Usage

There are three ways to get started with Artisan Engine.

<details>
<summary><strong>🐳 Option 1: Docker (Recommended)</strong></summary>

This is the easiest and recommended way to run Artisan Engine. It uses the pre-built image from the GitHub Container Registry.

**Prerequisites:**
* Docker installed.
* A GGUF-format model file downloaded to your machine.

**1. Prepare Your Model Directory**

Create a directory on your computer and place your downloaded `.gguf` model file inside it.

For example, you can use: [Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf). 

```bash
# For example:
mkdir my-local-model
mv ~/Downloads/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf ./my-local-model/
```

**2. Run the Artisan Engine Container**

This command will pull the latest image and start the server.

* **For Linux/macOS:**
    ```bash
    docker run -d --rm \
      -p 8000:8000 \
      -v "$(pwd)/my-local-model:/app/models" \
      -e ARTISAN_MODEL_PATH="/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf" \
      --name artisan-engine \
      ghcr.io/aafre/artisan-engine:latest
    ```

* **For Windows (PowerShell):**
    ```powershell
    docker run -d --rm `
      --gpus all `
      -p 8000:8000 `
      -v ${PWD}/my-local-model:/app/models `
      -e ARTISAN_MODEL_PATH="/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf" `
      -e ARTISAN_MODEL_LAZY_LOADING="false" `
      --name artisan-engine `
      ghcr.io/aafre/artisan-engine:latest
    ```

> **Note:** Replace `my-local-model` with the path to your model. 

The server will now be running at `http://localhost:8000`.

</details>

<details>
<summary><strong>🐍 Option 2: From PyPI (with pip)</strong></summary>

This method is for running the server directly on your machine without Docker.

**Prerequisites:**
* Python 3.12+ installed.
* A GGUF-format model file downloaded to your machine.

**1. Install the package:**
```bash
pip install artisan-engine
```

**2. Set the Model Path Environment Variable**

Artisan Engine needs to know where to find your model file.

* **For Linux/macOS:**
    ```bash
    export ARTISAN_MODEL_PATH="/path/to/your/model.gguf"
    ```

* **For Windows (PowerShell):**
    ```powershell
    $env:ARTISAN_MODEL_PATH = "C:\\path\\to\\your\\model.gguf"
    ```

**3. Run the Server**

Use the built-in CLI to start the server.
```bash
artisan serve
```

The server will now be running at `http://localhost:8000`.

</details>

<details>
<summary><strong>🧑‍💻 Option 3: From Source (for Developers)</strong></summary>

This method is for developers who want to contribute to Artisan Engine.

**1. Clone the repository:**
```bash
git clone https://github.com/aafre/artisan-engine.git
cd artisan-engine
```

**2. Install Dependencies:**
We recommend using [uv](https://docs.astral.sh/) for managing dependencies.
```bash
uv sync
```

**3. Configure Your Model:**
Copy the example environment file and edit it to point to your model.
```bash
cp .env.example .env
# Now, edit the .env file and set ARTISAN_MODEL_PATH
```

**4. Run the Development Server:**
Use the CLI to start the server
```bash
artisan serve
```
</details>

---

### Project Status & Roadmap

* [x] **v0.1.0: Guaranteed Structured Output**
* [ ] **v0.2.0: Full Function Calling / Tool Use**
* [ ] **v0.3.0: The Assistants API (Stateful Conversations)**
* [ ] **Future:** Integrated RAG, Expanded Backend Support (Ollama, vLLM)

---

### Contributing

Contributions are welcome! We have several issues flagged as `good first issue` that are perfect for getting started. Please see the [Issues Tab](https://github.com/aafre/artisan-engine/issues) to get involved.