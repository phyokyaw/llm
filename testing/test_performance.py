#!/usr/bin/env python3
"""
Enhanced performance testing script for vLLM servers with semantic similarity analysis.
Tests response times and semantic quality for llm1 and llm2 with various prompts.
"""

import time
import requests
import json
import statistics
from typing import List, Dict, Tuple, Optional
import argparse
from .test_fixtures import all_prompts
from .semantic_analyzer import SemanticAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPerformanceTester:
    """Enhanced performance tester with semantic analysis."""

    def __init__(self, semantic_analyzer: Optional[SemanticAnalyzer] = None):
        """
        Initialize the performance tester.

        Args:
            semantic_analyzer: Optional semantic analyzer instance
        """
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer()

        # Load reference data for expected responses
        self.reference_data = self._load_reference_data()

    def _load_reference_data(self) -> Dict[str, Dict[str, str]]:
        """Load reference conversation data for expected responses."""
        import pandas as pd

        try:
            df = pd.read_csv('reference_conversions.csv')
            reference_data = {}

            for _, row in df.iterrows():
                user_msg = str(row['user_message']).strip()
                response = str(row['response']).strip()

                if user_msg and response:
                    reference_data[user_msg] = response

            return reference_data
        except Exception as e:
            logger.warning(f"Failed to load reference data: {e}")
            return {}

    def _get_expected_response(self, prompt: str, category: str) -> str:
        """Get expected response for a given prompt."""
        # Try exact match first
        if prompt in self.reference_data:
            return self.reference_data[prompt]

        # Try to find similar prompt in reference data
        for ref_prompt, ref_response in self.reference_data.items():
            # Simple similarity check (you could make this more sophisticated)
            if len(set(prompt.split()) & set(ref_prompt.split())) > 0:
                return ref_response

        return ""

    def test_server_performance_with_semantics(
        self,
        server_name: str,
        model: str,
        port: int,
        prompts: List[str],
        categories: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, any]:
        """Test performance of a single server with semantic analysis."""

        print(f"\n🚀 Testing {server_name} (Port {port})")
        print(f"Model: {model}")
        print(f"Prompts: {len(prompts)}")
        print("-" * 50)

        url = f"http://localhost:{port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        response_times = []
        successful_requests = 0
        failed_requests = 0
        responses = []
        semantic_analyses = []

        for i, (prompt, category) in enumerate(zip(prompts, categories), 1):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            try:
                start_time = time.time()
                response = requests.post(
                    url, headers=headers, json=payload, timeout=30)
                end_time = time.time()

                response_time = end_time - start_time
                response_times.append(response_time)

                if response.status_code == 200:
                    successful_requests += 1
                    data = response.json()
                    tokens_used = data.get('usage', {}).get('total_tokens', 0)
                    tokens_per_second = tokens_used / response_time if response_time > 0 else 0

                    # Extract response content
                    response_content = ""
                    if 'choices' in data and len(data['choices']) > 0:
                        response_content = data['choices'][0]['message']['content']

                        # Check if response was truncated
                        finish_reason = data['choices'][0].get(
                            'finish_reason', '')
                        if finish_reason == 'length':
                            logger.warning(
                                f"Response truncated due to length limit for prompt {i}")
                        elif finish_reason == 'stop':
                            logger.info(
                                f"Response completed normally for prompt {i}")
                        else:
                            logger.info(
                                f"Response finish reason: {finish_reason} for prompt {i}")

                    # Get expected reference response
                    expected_response = self._get_expected_response(
                        prompt, category)

                    responses.append({
                        'prompt': prompt,
                        'actual_response': response_content,
                        'expected_response': expected_response,
                        'category': category
                    })

                    # Print input prompt and response
                    print(f"\n  📝 Prompt {i}:")
                    print(
                        f"     Input:  {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    print(
                        f"     Actual: {response_content[:100]}{'...' if len(response_content) > 100 else ''}")
                    if expected_response:
                        print(
                            f"     Expected: {expected_response[:100]}{'...' if len(expected_response) > 100 else ''}")

                    # Perform semantic analysis
                    if self.semantic_analyzer:
                        try:
                            analysis = self.semantic_analyzer.analyze_response_quality(
                                prompt, response_content, category
                            )
                            semantic_analyses.append(analysis)
                            quality_score = analysis['quality_score']

                            print(
                                f"  ⚡ {i:2d}. {response_time:.3f}s | {tokens_used:3d} tokens | {tokens_per_second:.1f} tok/s | Quality: {quality_score:.3f}")
                        except Exception as e:
                            logger.warning(
                                f"Semantic analysis failed for prompt {i}: {e}")
                            semantic_analyses.append(None)
                            print(
                                f"  ⚡ {i:2d}. {response_time:.3f}s | {tokens_used:3d} tokens | {tokens_per_second:.1f} tok/s | Quality: N/A")
                    else:
                        print(
                            f"  ⚡ {i:2d}. {response_time:.3f}s | {tokens_used:3d} tokens | {tokens_per_second:.1f} tok/s")

                else:
                    failed_requests += 1
                    expected_response = self._get_expected_response(
                        prompt, category)
                    responses.append({
                        'prompt': prompt,
                        'actual_response': "",
                        'expected_response': expected_response,
                        'category': category
                    })
                    semantic_analyses.append(None)
                    print(f"  {i:2d}. FAILED ({response.status_code})")

            except requests.exceptions.RequestException as e:
                failed_requests += 1
                expected_response = self._get_expected_response(
                    prompt, category)
                responses.append({
                    'prompt': prompt,
                    'actual_response': "",
                    'expected_response': expected_response,
                    'category': category
                })
                semantic_analyses.append(None)
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
                'std_deviation': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'responses': responses,
                'semantic_analyses': semantic_analyses
            }

            # Add semantic quality statistics
            if semantic_analyses and any(analysis is not None for analysis in semantic_analyses):
                quality_scores = [analysis['quality_score']
                                  for analysis in semantic_analyses if analysis is not None]
                if quality_scores:
                    stats.update({
                        'avg_quality_score': statistics.mean(quality_scores),
                        'median_quality_score': statistics.median(quality_scores),
                        'min_quality_score': min(quality_scores),
                        'max_quality_score': max(quality_scores),
                        'quality_std_deviation': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                        'high_quality_responses': sum(1 for score in quality_scores if score > 0.7),
                        'medium_quality_responses': sum(1 for score in quality_scores if 0.4 <= score <= 0.7),
                        'low_quality_responses': sum(1 for score in quality_scores if score < 0.4)
                    })
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
                'std_deviation': 0,
                'responses': responses,
                'semantic_analyses': semantic_analyses
            }

        return stats

    def print_performance_summary(self, server_name: str, stats: Dict[str, any]):
        """Print a comprehensive summary of performance statistics."""
        print(f"\n📊 {server_name} Performance Summary:")
        print("=" * 60)
        print(f"Total Requests:     {stats['total_requests']}")
        print(f"Successful:         {stats['successful_requests']}")
        print(f"Failed:             {stats['failed_requests']}")
        print(f"Success Rate:       {stats['success_rate']:.1f}%")
        print(f"Average Time:       {stats['avg_response_time']:.3f}s")
        print(f"Median Time:        {stats['median_response_time']:.3f}s")
        print(f"Min Time:           {stats['min_response_time']:.3f}s")
        print(f"Max Time:           {stats['max_response_time']:.3f}s")
        print(f"Std Deviation:      {stats['std_deviation']:.3f}s")

        # Semantic quality statistics
        if 'avg_quality_score' in stats:
            print(f"\n🎯 Semantic Quality Analysis:")
            print(f"Average Quality:    {stats['avg_quality_score']:.3f}")
            print(f"Median Quality:     {stats['median_quality_score']:.3f}")
            print(f"Min Quality:        {stats['min_quality_score']:.3f}")
            print(f"Max Quality:        {stats['max_quality_score']:.3f}")
            print(f"Quality Std Dev:    {stats['quality_std_deviation']:.3f}")
            print(f"High Quality (>0.7): {stats['high_quality_responses']}")
            print(f"Medium Quality:     {stats['medium_quality_responses']}")
            print(f"Low Quality (<0.4):  {stats['low_quality_responses']}")

    def compare_servers(self, stats1: Dict[str, any], stats2: Dict[str, any]):
        """Compare performance between two servers including semantic analysis."""
        print(f"\n⚡ Performance Comparison:")
        print("=" * 60)

        print(f"{'Metric':<25} {'LLM1':<12} {'LLM2':<12} {'Winner':<8}")
        print("-" * 60)

        # Compare average response time
        avg1, avg2 = stats1['avg_response_time'], stats2['avg_response_time']
        winner_avg = "LLM1" if avg1 < avg2 else "LLM2" if avg2 < avg1 else "Tie"
        print(
            f"{'Avg Response Time':<25} {avg1:.3f}s{'':<6} {avg2:.3f}s{'':<6} {winner_avg:<8}")

        # Compare success rate
        sr1, sr2 = stats1['success_rate'], stats2['success_rate']
        winner_sr = "LLM1" if sr1 > sr2 else "LLM2" if sr2 > sr1 else "Tie"
        print(
            f"{'Success Rate':<25} {sr1:.1f}%{'':<7} {sr2:.1f}%{'':<7} {winner_sr:<8}")

        # Compare consistency (lower std dev is better)
        std1, std2 = stats1['std_deviation'], stats2['std_deviation']
        winner_std = "LLM1" if std1 < std2 else "LLM2" if std2 < std1 else "Tie"
        print(
            f"{'Consistency':<25} {std1:.3f}s{'':<6} {std2:.3f}s{'':<6} {winner_std:<8}")

        # Compare semantic quality if available
        if 'avg_quality_score' in stats1 and 'avg_quality_score' in stats2:
            q1, q2 = stats1['avg_quality_score'], stats2['avg_quality_score']
            winner_q = "LLM1" if q1 > q2 else "LLM2" if q2 > q1 else "Tie"
            print(
                f"{'Avg Quality Score':<25} {q1:.3f}{'':<8} {q2:.3f}{'':<8} {winner_q:<8}")

            hq1, hq2 = stats1['high_quality_responses'], stats2['high_quality_responses']
            winner_hq = "LLM1" if hq1 > hq2 else "LLM2" if hq2 > hq1 else "Tie"
            print(
                f"{'High Quality Count':<25} {hq1}{'':<9} {hq2}{'':<9} {winner_hq:<8}")

    def generate_detailed_report(self, results: Dict[str, Dict]) -> Dict[str, any]:
        """Generate a detailed report with semantic analysis."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'servers_tested': list(results.keys()),
            'server_results': results,
            'summary': {}
        }

        # Calculate overall summary
        if len(results) >= 2:
            server_names = list(results.keys())
            stats1, stats2 = results[server_names[0]], results[server_names[1]]

            report['summary'] = {
                'performance_winner': self._determine_performance_winner(stats1, stats2),
                'quality_winner': self._determine_quality_winner(stats1, stats2),
                'overall_recommendation': self._generate_recommendation(stats1, stats2)
            }

        return report

    def _determine_performance_winner(self, stats1: Dict, stats2: Dict) -> str:
        """Determine performance winner based on multiple metrics."""
        metrics = [
            ('avg_response_time', False),  # Lower is better
            ('success_rate', True),        # Higher is better
            ('std_deviation', False)       # Lower is better
        ]

        score1, score2 = 0, 0

        for metric, higher_is_better in metrics:
            if metric in stats1 and metric in stats2:
                val1, val2 = stats1[metric], stats2[metric]
                if higher_is_better:
                    if val1 > val2:
                        score1 += 1
                    elif val2 > val1:
                        score2 += 1
                else:
                    if val1 < val2:
                        score1 += 1
                    elif val2 < val1:
                        score2 += 1

        if score1 > score2:
            return "Server 1"
        elif score2 > score1:
            return "Server 2"
        else:
            return "Tie"

    def _determine_quality_winner(self, stats1: Dict, stats2: Dict) -> str:
        """Determine quality winner based on semantic analysis."""
        if 'avg_quality_score' not in stats1 or 'avg_quality_score' not in stats2:
            return "Not Available"

        q1, q2 = stats1['avg_quality_score'], stats2['avg_quality_score']

        if q1 > q2:
            return "Server 1"
        elif q2 > q1:
            return "Server 2"
        else:
            return "Tie"

    def _generate_recommendation(self, stats1: Dict, stats2: Dict) -> str:
        """Generate overall recommendation."""
        perf_winner = self._determine_performance_winner(stats1, stats2)
        qual_winner = self._determine_quality_winner(stats1, stats2)

        if perf_winner == qual_winner and perf_winner != "Tie":
            return f"Recommend {perf_winner} for both performance and quality"
        elif perf_winner != "Tie" and qual_winner != "Tie":
            return f"Trade-off: {perf_winner} for performance, {qual_winner} for quality"
        else:
            return "Both servers perform similarly"


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced vLLM server performance testing with semantic analysis")
    parser.add_argument("--max-tokens", type=int,
                        default=500, help="Max tokens per response")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Temperature for generation")
    parser.add_argument("--server1-only", action="store_true",
                        help="Test only server 1")
    parser.add_argument("--server2-only", action="store_true",
                        help="Test only server 2")
    parser.add_argument("--prompt-category", type=str, default="auto",
                        choices=["auto", "general", "myanmar_internet_outage", "myanmar_billing_payment",
                                 "myanmar_technical_support", "myanmar_product_inquiry",
                                 "myanmar_customer_service", "myanmar_mixed_language",
                                 "myanmar_combined", "technical", "creative", "customer_service"],
                        help="Prompt category to use for testing")
    parser.add_argument("--enable-semantics", action="store_true", default=True,
                        help="Enable semantic similarity analysis")
    parser.add_argument("--save-report", type=str, default="performance_report.json",
                        help="Save detailed report to JSON file")

    args = parser.parse_args()

    # Initialize semantic analyzer if enabled
    semantic_analyzer = None
    if args.enable_semantics:
        print("🧠 Initializing semantic analyzer...")
        semantic_analyzer = SemanticAnalyzer()
        semantic_analyzer.load_reference_from_csv('reference_conversions.csv')
        print("✅ Semantic analyzer ready!")

    # Initialize tester
    tester = EnhancedPerformanceTester(semantic_analyzer)

    # Select prompts based on category and server
    if args.prompt_category == "auto":
        # Auto-select based on server
        llm1_prompts = all_prompts["general"]
        llm2_prompts = all_prompts["myanmar_combined"]

        # Create corresponding categories
        llm1_categories = ["general"] * len(llm1_prompts)
        llm2_categories = ["myanmar_combined"] * len(llm2_prompts)
    else:
        # Use specified category for both servers
        llm1_prompts = all_prompts.get(
            args.prompt_category, all_prompts["general"])
        llm2_prompts = all_prompts.get(
            args.prompt_category, all_prompts["myanmar_combined"])

        llm1_categories = [args.prompt_category] * len(llm1_prompts)
        llm2_categories = [args.prompt_category] * len(llm2_prompts)

    print("🧪 Enhanced vLLM Performance Testing with Semantic Analysis")
    print("=" * 70)
    print(f"Prompt Category: {args.prompt_category}")
    print(f"LLM1 Prompts: {len(llm1_prompts)}")
    print(f"LLM2 Prompts: {len(llm2_prompts)}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(
        f"Semantic Analysis: {'Enabled' if args.enable_semantics else 'Disabled'}")

    # Server configurations
    servers = [
        {
            "name": "LLM1 (Gemma)",
            "model": "aisingapore/Gemma-SEA-LION-v3-9B-IT",
            "port": 8000,
            "prompts": llm1_prompts,
            "categories": llm1_categories
        },
        {
            "name": "LLM2 (fr_sealion)",
            "model": "NanEi/fr_sealion_merge_bot_v1-5",
            "port": 8001,
            "prompts": llm2_prompts,
            "categories": llm2_categories
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
        stats = tester.test_server_performance_with_semantics(
            server["name"],
            server["model"],
            server["port"],
            server["prompts"],
            server["categories"],
            args.max_tokens,
            args.temperature
        )
        results[server["name"]] = stats
        tester.print_performance_summary(server["name"], stats)

    # Compare results if both servers were tested
    if len(results) == 2:
        server_names = list(results.keys())
        tester.compare_servers(
            results[server_names[0]], results[server_names[1]])

    # Generate and save detailed report
    if args.save_report:
        detailed_report = tester.generate_detailed_report(results)
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Detailed report saved to {args.save_report}")

    print(f"\n✅ Enhanced performance testing completed!")


if __name__ == "__main__":
    main()
