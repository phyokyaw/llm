#!/usr/bin/env python3
"""
Quick performance test for vLLM servers.
Simple script to test response times with a few prompts.
"""

import time
import requests
import json
from test_fixtures import all_prompts


def quick_test(server_name: str, model: str, port: int, prompts: list, max_tokens: int = 50):
    """Quick performance test for a single server."""
    
    print(f"\n🚀 Quick Test: {server_name}")
    print(f"Model: {model}")
    print(f"Port: {port}")
    print("-" * 40)
    
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    total_time = 0
    successful = 0
    
    for i, prompt in enumerate(prompts, 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            total_time += response_time
            
            if response.status_code == 200:
                successful += 1
                data = response.json()
                tokens = data.get('usage', {}).get('total_tokens', 0)
                tokens_per_sec = tokens / response_time if response_time > 0 else 0
                
                print(f"  {i}. {response_time:.3f}s | {tokens} tokens | {tokens_per_sec:.1f} tok/s")
            else:
                print(f"  {i}. FAILED ({response.status_code})")
                
        except Exception as e:
            print(f"  {i}. ERROR: {e}")
    
    if successful > 0:
        avg_time = total_time / successful
        print(f"\n📊 Summary:")
        print(f"  Successful: {successful}/{len(prompts)}")
        print(f"  Average time: {avg_time:.3f}s")
    else:
        print(f"\n❌ No successful requests")


def main():
    print("⚡ Quick Performance Test")
    print("=" * 40)
    
    # Test LLM1 (Gemma) with general prompts
    quick_test(
        "LLM1 (Gemma)",
        "aisingapore/Gemma-SEA-LION-v3-9B-IT",
        8000,
        all_prompts["general"][:3],  # First 3 general prompts
        max_tokens=50
    )
    
    # Test LLM2 (fr_sealion) with Myanmar ISP prompts
    quick_test(
        "LLM2 (fr_sealion)",
        "NanEi/fr_sealion_merge_bot_v1-5",
        8001,
        all_prompts["myanmar_isp_product_prompts"][:3],  # First 3 Myanmar ISP prompts
        max_tokens=50
    )
    
    print(f"\n✅ Quick test completed!")


if __name__ == "__main__":
    main()
