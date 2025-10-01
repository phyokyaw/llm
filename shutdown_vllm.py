#!/usr/bin/env python3
"""
Graceful shutdown script for vLLM servers.
This script finds and terminates all vLLM processes gracefully.
"""

import os
import sys
import signal
import subprocess
import time
from typing import List, Optional


def find_vllm_processes() -> List[dict]:
    """Find all running vLLM processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'vllm' in line.lower() and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = int(parts[1])
                    cmd = ' '.join(parts[10:])
                    processes.append({
                        'pid': pid,
                        'cmd': cmd,
                        'line': line.strip()
                    })
        
        return processes
    except subprocess.CalledProcessError as e:
        print(f"Error finding processes: {e}")
        return []


def shutdown_process(pid: int, timeout: int = 10) -> bool:
    """Gracefully shutdown a process by PID."""
    try:
        # First try SIGTERM (graceful)
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
        
        # Wait for process to exit
        for i in range(timeout):
            try:
                os.kill(pid, 0)  # Check if process exists
                time.sleep(1)
            except ProcessLookupError:
                print(f"Process {pid} exited gracefully")
                return True
        
        # If still running, try SIGKILL (force)
        print(f"Process {pid} didn't exit, sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(2)
        
        try:
            os.kill(pid, 0)
            print(f"Failed to kill process {pid}")
            return False
        except ProcessLookupError:
            print(f"Process {pid} killed forcefully")
            return True
            
    except ProcessLookupError:
        print(f"Process {pid} already exited")
        return True
    except PermissionError:
        print(f"Permission denied to kill process {pid}")
        return False
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False


def check_gpu_memory():
    """Check GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("\nGPU Memory Status:")
        print("GPU | Used (MB) | Total (MB) | Usage %")
        print("-" * 40)
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id = parts[0]
                    used = int(parts[1])
                    total = int(parts[2])
                    usage = (used / total) * 100
                    print(f" {gpu_id}  | {used:8} | {total:9} | {usage:6.1f}%")
        
    except subprocess.CalledProcessError:
        print("Could not check GPU memory (nvidia-smi not available)")
    except Exception as e:
        print(f"Error checking GPU memory: {e}")


def main():
    """Main shutdown function."""
    print("🛑 vLLM Server Shutdown Script")
    print("=" * 50)
    
    # Find all vLLM processes
    processes = find_vllm_processes()
    
    if not processes:
        print("✅ No vLLM processes found running")
        check_gpu_memory()
        return
    
    print(f"Found {len(processes)} vLLM process(es):")
    for proc in processes:
        print(f"  PID {proc['pid']}: {proc['cmd'][:80]}...")
    
    print("\n🔄 Shutting down processes...")
    
    # Shutdown processes
    success_count = 0
    for proc in processes:
        if shutdown_process(proc['pid']):
            success_count += 1
    
    print(f"\n📊 Shutdown Summary:")
    print(f"  Total processes: {len(processes)}")
    print(f"  Successfully stopped: {success_count}")
    print(f"  Failed to stop: {len(processes) - success_count}")
    
    # Wait a moment for cleanup
    time.sleep(2)
    
    # Check if any processes are still running
    remaining = find_vllm_processes()
    if remaining:
        print(f"\n⚠️  {len(remaining)} process(es) still running:")
        for proc in remaining:
            print(f"  PID {proc['pid']}: {proc['cmd'][:80]}...")
    else:
        print("\n✅ All vLLM processes stopped successfully")
    
    # Check GPU memory
    check_gpu_memory()
    
    print("\n🎉 Shutdown complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutdown interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
