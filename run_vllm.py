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
    dtype: str = "float16"
    host: str = "0.0.0.0"
    max_model_len: int = 8192


def read_server(prefix: str, default_name: str) -> Optional[ServerConfig]:
    def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(f"{prefix}_{key}", default)

    model = getenv("MODEL")
    if not model:
        return None

    port_str = getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        raise SystemExit(f"Invalid {prefix}_PORT: {port_str}")

    tp_str = getenv("TP", getenv("TENSOR_PARALLEL_SIZE", "1"))
    try:
        tp = int(tp_str)
    except ValueError:
        raise SystemExit(f"Invalid {prefix}_TP: {tp_str}")

    dtype = getenv("DTYPE", "float16")
    host = getenv("HOST", "0.0.0.0")
    max_len = int(getenv("MAX_MODEL_LEN", "8192"))

    return ServerConfig(
        name=getenv("NAME", default_name),
        model=model,
        port=port,
        tensor_parallel_size=tp,
        dtype=dtype,
        host=host,
        max_model_len=max_len,
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

    console.print(f"Starting vLLM [{cfg.name}] on port {cfg.port} for model {cfg.model}")
    stdout = open(log_path, "a")
    stderr = subprocess.STDOUT
    proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
    return proc


def main() -> None:
    load_dotenv(override=True)

    # Enable flash attention v2 if present; harmless otherwise.
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_USE_V2", "1")
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

    # Optional: model cache + HF token
    model_cache = os.getenv("VM_MODEL_CACHE")
    if model_cache:
        env["HF_HOME"] = model_cache
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    s1 = read_server("VM1", "vm1")
    s2 = read_server("VM2", "vm2")

    servers = [s for s in (s1, s2) if s is not None]
    if not servers:
        console.print("[red]No servers configured. Set VM1_MODEL/VM2_MODEL in env.[/red]")
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


