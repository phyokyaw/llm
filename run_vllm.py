import os
import sys
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, List

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()


@dataclass
class ServerConfig:
    name: str
    model: str
    port: int
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    host: str = "0.0.0.0"
    max_model_len: int = 8192
    cuda_visible_devices: Optional[str] = None


def read_server(index: int, default_name: str) -> Optional[ServerConfig]:
    # New schema: LLM_*{index}
    def get_new(name: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(f"LLM_{name}{index}", default)

    # Backward-compat: VM{index}_*
    def get_old(name: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(f"VM{index}_{name}", default)

    def choose(new_key: str, old_key: str, default: Optional[str] = None) -> Optional[str]:
        val = get_new(new_key)
        if val is not None and val != "":
            return val
        return get_old(old_key, default)

    model = choose("MODEL", "MODEL")
    if not model:
        return None

    default_port = "8000" if index == 1 else "8001"
    port_str = choose("PORT", "PORT", default_port)
    try:
        port = int(port_str)
    except ValueError:
        raise SystemExit(f"Invalid port for server {index}: {port_str}")

    tp_str = choose("TP", "TP", None) or choose("TENSOR_PARALLEL_SIZE", "TENSOR_PARALLEL_SIZE", "1")
    try:
        tp = int(tp_str)
    except ValueError:
        raise SystemExit(f"Invalid TP for server {index}: {tp_str}")

    # Use bfloat16 for Gemma models, float16 for others
    default_dtype = "bfloat16" if "gemma" in model.lower() else "float16"
    dtype = choose("DTYPE", "DTYPE", default_dtype)
    host = choose("HOST", "HOST", "0.0.0.0")
    # Use smaller max_model_len for sentence-transformers models
    default_max_len = 128 if "sentence-transformers" in model.lower() else 8192
    max_len = int(choose("MAX_MODEL_LEN", "MAX_MODEL_LEN", str(default_max_len)))
    cuda = get_new("CUDA_VISIBLE_DEVICES", None) or get_old("CUDA_VISIBLE_DEVICES", None)

    return ServerConfig(
        name=choose("NAME", "NAME", default_name),
        model=model,
        port=port,
        tensor_parallel_size=tp,
        dtype=dtype,
        host=host,
        max_model_len=max_len,
        cuda_visible_devices=cuda,
    )


def start_server(cfg: ServerConfig, env: dict) -> subprocess.Popen:
    log_dir = os.path.expanduser("~/vllm_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vllm_{cfg.name or cfg.port}.log")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model,
        "--port",
        str(cfg.port),
        "--host",
        cfg.host,
        "--dtype",
        cfg.dtype,
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
        "--max-model-len",
        str(cfg.max_model_len),
    ]

    run_env = env.copy()
    if cfg.cuda_visible_devices:
        run_env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

    gpu_note = f" (CUDA_VISIBLE_DEVICES={cfg.cuda_visible_devices})" if cfg.cuda_visible_devices else ""
    console.print(f"Starting vLLM [{cfg.name}] on port {cfg.port} for model {cfg.model}{gpu_note}")
    stdout = open(log_path, "a")
    stderr = subprocess.STDOUT
    proc = subprocess.Popen(cmd, env=run_env, stdout=stdout, stderr=stderr)
    return proc


def main() -> None:
    load_dotenv(override=True)

    # Enable flash attention v2 if present; harmless otherwise.
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_USE_V2", "1")
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    # Optional: model cache + HF token
    model_cache = os.getenv("LLM_MODEL_CACHE") or os.getenv("VM_MODEL_CACHE")
    if model_cache:
        env["HF_HOME"] = model_cache
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    s1 = read_server(1, "llm1")
    s2 = read_server(2, "llm2")
    s3 = read_server(3, "llm3")

    servers = [s for s in (s1, s2, s3) if s is not None]
    if not servers:
        console.print("[red]No servers configured. Set LLM_MODEL1/LLM_MODEL2/LLM_MODEL3 in env.[/red]")
        sys.exit(1)

    table = Table(title="Local vLLM Servers")
    for col in ("Name", "Port", "Model", "TP", "Dtype"):
        table.add_column(col)
    for s in servers:
        table.add_row(s.name, str(s.port), s.model, str(s.tensor_parallel_size), s.dtype)
    console.print(table)

    procs = [start_server(s, env) for s in servers]

    console.print("Waiting 3s for servers to initialize...")
    time.sleep(3)
    for s in servers:
        console.print(f"- http://{s.host}:{s.port}/v1")

    # Keep running until any server exits
    try:
        while True:
            for idx, p in enumerate(procs):
                ret = p.poll()
                if ret is not None:
                    console.print(f"[red]Server {servers[idx].name} exited with {ret}[/red]")
                    sys.exit(ret)
            time.sleep(2)
    except KeyboardInterrupt:
        console.print("Stopping servers...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()


