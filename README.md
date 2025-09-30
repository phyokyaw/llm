# llm

Utilities to run two Hugging Face models behind vLLM OpenAI-compatible servers on a single VM with two RTX 3090 GPUs. Intended to host two ~18GB models concurrently on separate ports.

## Prerequisites

- Single VM with two RTX 3090 GPUs and recent NVIDIA drivers/CUDA
- Inbound TCP access to the chosen ports (default 8000/8001)

## Setup

1) Copy the env template and fill in values (two ~18GB models, separate ports):

```bash
cp sample.env .env
$EDITOR .env
```

Set at least `LLM_MODEL1`, `LLM_MODEL2`, `LLM_MODEL3`. Optionally set `HF_TOKEN` for gated models. You can pin each server to a GPU via `LLM_CUDA_VISIBLE_DEVICES1` and `LLM_CUDA_VISIBLE_DEVICES2`.

2) Install Poetry deps on the VM:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

## Run on the VM (no SSH needed)

Run directly on the VM:

```bash
poetry run run-vllm
```

Set env vars to choose models/ports. Example:

```bash
export LLM_MODEL1=aisingapore/Gemma-SEA-LION-v3-9B-IT
export LLM_PORT1=8000
export LLM_CUDA_VISIBLE_DEVICES1=0
export LLM_MODEL2=NanEi/fr_sealion_merge_bot_v1-5
export LLM_PORT2=8001
export LLM_CUDA_VISIBLE_DEVICES2=1
export LLM_MODEL3=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export LLM_PORT3=8002
export HF_TOKEN= # if needed for gated models
poetry run run-vllm
```

Endpoints will be available at:

- Server1: `http://0.0.0.0:${LLM_PORT1:-8000}/v1`
- Server2: `http://0.0.0.0:${LLM_PORT2:-8001}/v1`
- Server3: `http://0.0.0.0:${LLM_PORT3:-8002}/v1`

Notes:
- For performance, set `VM_MODEL_CACHE` in `.env` to a persistent path.

## Verify

From your machine:

```bash
curl http://<vm-ip>:8000/v1/models | jq
curl http://<vm-ip>:8001/v1/models | jq
curl http://<vm-ip>:8002/v1/models | jq
```

## Notes

- The runner starts three vLLM servers in the foreground; logs are written to `~/vllm_logs`.
- Set `VM_MODEL_CACHE` to control where models are cached on the VM.
- With two GPUs, you can pin each server to a GPU via `CUDA_VISIBLE_DEVICES`. Example: open two shells and export `CUDA_VISIBLE_DEVICES=0` for the first server and `CUDA_VISIBLE_DEVICES=1` for the second, or run the script twice with different env.
- For single-GPU-per-server, keep `--tensor-parallel-size` at 1.

FR Chatbot llm.
