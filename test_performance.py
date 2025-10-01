#!/usr/bin/env python3
"""
Performance testing script for vLLM servers.
Tests response times for llm1 and llm2 with various prompts.
"""

import time
import requests
import json
import statistics
from typing import List, Dict, Tuple
import argparse
from test_fixtures import all_prompts


def test_server_performance(
    server_name: str,
    model: str,
    port: int,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.7
) -> Dict[str, float]:
    """Test performance of a single server with multiple prompts."""
    
    print(f"\n🚀 Testing {server_name} (Port {port})")
    print(f"Model: {model}")
    print(f"Prompts: {len(prompts)}")
    print("-" * 50)
    
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    response_times = []
    successful_requests = 0
    failed_requests = 0
    
    for i, prompt in enumerate(prompts, 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if response.status_code == 200:
                successful_requests += 1
                data = response.json()
                tokens_used = data.get('usage', {}).get('total_tokens', 0)
                tokens_per_second = tokens_used / response_time if response_time > 0 else 0
                
                print(f"  {i:2d}. {response_time:.3f}s | {tokens_used:3d} tokens | {tokens_per_second:.1f} tok/s")
            else:
                failed_requests += 1
                print(f"  {i:2d}. FAILED ({response.status_code})")
                
        except requests.exceptions.RequestException as e:
            failed_requests += 1
            print(f"  {i:2d}. ERROR: {e}")
    
    # Calculate statistics
    if response_times:
        stats = {
            'total_requests': len(prompts),
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': successful_requests / len(prompts) * 100,
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'std_deviation': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    else:
        stats = {
            'total_requests': len(prompts),
            'successful_requests': 0,
            'failed_requests': failed_requests,
            'success_rate': 0,
            'avg_response_time': 0,
            'median_response_time': 0,
            'min_response_time': 0,
            'max_response_time': 0,
            'std_deviation': 0
        }
    
    return stats


def print_performance_summary(server_name: str, stats: Dict[str, float]):
    """Print a summary of performance statistics."""
    print(f"\n📊 {server_name} Performance Summary:")
    print("=" * 50)
    print(f"Total Requests:     {stats['total_requests']}")
    print(f"Successful:         {stats['successful_requests']}")
    print(f"Failed:             {stats['failed_requests']}")
    print(f"Success Rate:       {stats['success_rate']:.1f}%")
    print(f"Average Time:       {stats['avg_response_time']:.3f}s")
    print(f"Median Time:        {stats['median_response_time']:.3f}s")
    print(f"Min Time:           {stats['min_response_time']:.3f}s")
    print(f"Max Time:           {stats['max_response_time']:.3f}s")
    print(f"Std Deviation:      {stats['std_deviation']:.3f}s")


def compare_servers(stats1: Dict[str, float], stats2: Dict[str, float]):
    """Compare performance between two servers."""
    print(f"\n⚡ Performance Comparison:")
    print("=" * 50)
    
    print(f"{'Metric':<20} {'LLM1':<12} {'LLM2':<12} {'Winner':<8}")
    print("-" * 50)
    
    # Compare average response time
    avg1, avg2 = stats1['avg_response_time'], stats2['avg_response_time']
    winner_avg = "LLM1" if avg1 < avg2 else "LLM2" if avg2 < avg1 else "Tie"
    print(f"{'Avg Response Time':<20} {avg1:.3f}s{'':<6} {avg2:.3f}s{'':<6} {winner_avg:<8}")
    
    # Compare success rate
    sr1, sr2 = stats1['success_rate'], stats2['success_rate']
    winner_sr = "LLM1" if sr1 > sr2 else "LLM2" if sr2 > sr1 else "Tie"
    print(f"{'Success Rate':<20} {sr1:.1f}%{'':<7} {sr2:.1f}%{'':<7} {winner_sr:<8}")
    
    # Compare consistency (lower std dev is better)
    std1, std2 = stats1['std_deviation'], stats2['std_deviation']
    winner_std = "LLM1" if std1 < std2 else "LLM2" if std2 < std1 else "Tie"
    print(f"{'Consistency':<20} {std1:.3f}s{'':<6} {std2:.3f}s{'':<6} {winner_std:<8}")


def main():
    parser = argparse.ArgumentParser(description="Test vLLM server performance")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--server1-only", action="store_true", help="Test only server 1")
    parser.add_argument("--server2-only", action="store_true", help="Test only server 2")
    parser.add_argument("--prompt-category", type=str, default="auto", 
                       choices=["auto", "general", "myanmar_isp_products", "myanmar_outage_complaints", 
                               "myanmar_combined", "technical", "creative", "customer_service"],
                       help="Prompt category to use for testing")
    
    args = parser.parse_args()
    
    # Select prompts based on category and server
    if args.prompt_category == "auto":
        # Auto-select based on server
        llm1_prompts = all_prompts["general"]
        llm2_prompts = all_prompts["myanmar_combined"]  # ISP products + outage complaints
    else:
        # Use specified category for both servers
        llm1_prompts = all_prompts.get(args.prompt_category, all_prompts["general"])
        llm2_prompts = all_prompts.get(args.prompt_category, all_prompts["myanmar_combined"])
    
    print("🧪 vLLM Performance Testing")
    print("=" * 50)
    print(f"Prompt Category: {args.prompt_category}")
    print(f"LLM1 Prompts: {len(llm1_prompts)}")
    print(f"LLM2 Prompts: {len(llm2_prompts)}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    
    # Server configurations
    servers = [
        {
            "name": "LLM1 (Gemma)",
            "model": "aisingapore/Gemma-SEA-LION-v3-9B-IT",
            "port": 8000,
            "prompts": llm1_prompts
        },
        {
            "name": "LLM2 (fr_sealion)",
            "model": "NanEi/fr_sealion_merge_bot_v1-5", 
            "port": 8001,
            "prompts": llm2_prompts
        }
    ]
    
    results = {}
    
    # Test servers based on arguments
    if args.server1_only:
        servers_to_test = [servers[0]]
    elif args.server2_only:
        servers_to_test = [servers[1]]
    else:
        servers_to_test = servers
    
    for server in servers_to_test:
        stats = test_server_performance(
            server["name"],
            server["model"],
            server["port"],
            server["prompts"],
            args.max_tokens,
            args.temperature
        )
        results[server["name"]] = stats
        print_performance_summary(server["name"], stats)
    
    # Compare results if both servers were tested
    if len(results) == 2:
        compare_servers(results["LLM1 (Gemma)"], results["LLM2 (fr_sealion)"])
    
    print(f"\n✅ Performance testing completed!")


if __name__ == "__main__":
    main()
