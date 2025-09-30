# llm

Utilities to deploy two Hugging Face models behind vLLM OpenAI-compatible servers on two separate RTX 3090 VMs.

## Prerequisites

- Two cloud VMs with NVIDIA RTX 3090 GPUs and recent CUDA drivers
- Inbound TCP access to the ports you choose (default 8000/8001)
- SSH key-based access

## Setup

1. Install Poetry and dependencies:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

2. Copy the env template and fill in your values:

```bash
cp .env.example .env
$EDITOR .env
```

At minimum set `VM1_HOST`, `VM1_MODEL`, `VM2_HOST`, `VM2_MODEL`. Optionally set `HF_TOKEN` if the models require auth.

## Run on the GPU VM (no SSH)

On the GPU VM where the models will run:

```bash
poetry run run-vllm
```

Set env vars to choose models/ports. Example:

```bash
export VM1_MODEL=meta-llama/Llama-3.1-8B-Instruct
export VM1_PORT=8000
export VM2_MODEL=mistralai/Mistral-7B-Instruct-v0.3
export VM2_PORT=8001
export HF_TOKEN= # if needed for gated models
poetry run run-vllm
```

Endpoints will be available at:

- VM1: `http://0.0.0.0:${VM1_PORT:-8000}/v1`
- VM2: `http://0.0.0.0:${VM2_PORT:-8001}/v1`

## Verify

From your machine:

```bash
curl http://<vm1-host>:8000/v1/models | jq
```

## Notes

- The script uses `nohup` to background the vLLM server and writes logs under `~/vllm_logs` on each VM.
- You can set `VMx_CONDA_ENV` or `VMx_VENV_PATH` to reuse an existing environment; otherwise a lightweight venv is created.
- Set `VMx_MODEL_CACHE` to control where models are cached on the VM.
- For single 3090 GPUs, keep `--tensor-parallel-size` at 1.

FR Chatbot llm.
