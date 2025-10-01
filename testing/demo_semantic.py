#!/usr/bin/env python3
"""
Demo script for semantic similarity analysis.
Shows how to use the semantic analyzer with Myanmar conversation data.
"""

import json
from .semantic_analyzer import SemanticAnalyzer

def demo_semantic_analysis():
    """Demonstrate semantic analysis functionality."""
    
    print("🧠 Semantic Similarity Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer
    print("Initializing semantic analyzer...")
    analyzer = SemanticAnalyzer()
    
    # Load reference data from CSV
    print("Loading reference conversations from CSV...")
    analyzer.load_reference_from_csv('reference_conversions.csv')
    
    print(f"✅ Loaded {len(analyzer.reference_texts)} categories")
    for category, texts in analyzer.reference_texts.items():
        print(f"  - {category}: {len(texts)} reference texts")
    
    # Demo queries
    demo_queries = [
        {
            "text": "လိုင်းမကောင်းပါဘူး ဘယ်လိုဖြစ်နေတာလဲဗျ",
            "category": "myanmar_internet_outage",
            "description": "Internet outage complaint"
        },
        {
            "text": "ဘေလ်ဆောင်ချင်လို့ပါ",
            "category": "myanmar_billing_payment", 
            "description": "Billing inquiry"
        },
        {
            "text": "wifi ချိတ်မရလို့ပါ",
            "category": "myanmar_technical_support",
            "description": "WiFi connection issue"
        },
        {
            "text": "ပက်ကေ့ဂ်တွေက ဘယ်လိုမျိုးတွေရှိလဲ",
            "category": "myanmar_product_inquiry",
            "description": "Product inquiry"
        }
    ]
    
    print("\n🔍 Analyzing demo queries...")
    print("-" * 50)
    
    for i, query_info in enumerate(demo_queries, 1):
        query_text = query_info["text"]
        category = query_info["category"]
        description = query_info["description"]
        
        print(f"\n{i}. {description}")
        print(f"   Query: {query_text}")
        print(f"   Category: {category}")
        
        # Calculate similarity
        similarity_result = analyzer.calculate_similarity(query_text, category)
        
        if 'error' not in similarity_result:
            print(f"   Max Similarity: {similarity_result['max_similarity']:.3f}")
            print(f"   Mean Similarity: {similarity_result['mean_similarity']:.3f}")
            
            # Show top similar texts
            print("   Top Similar References:")
            for j, top_match in enumerate(similarity_result['top_similarities'][:3], 1):
                print(f"     {j}. {top_match['similarity']:.3f} - {top_match['text'][:50]}...")
        else:
            print(f"   Error: {similarity_result['error']}")
    
    # Demo response quality analysis
    print("\n🎯 Response Quality Analysis Demo")
    print("-" * 50)
    
    demo_responses = [
        {
            "query": "လိုင်းမကောင်းပါဘူး",
            "response": "အဆင်မပြေမှုများအတွက်တောင်းပန်ပါတယ်။ Customer ID ကိုပေးပို့ပေးပါခင်ဗျာ။",
            "category": "myanmar_internet_outage"
        },
        {
            "query": "ဘေလ်ဆောင်ချင်လို့ပါ",
            "response": "Mobile Banking ဖြင့်ပေးဆောင်နည်းများကိုအောက်ပါ Link တွင်ကြည့်နိုင်ပါသည်။",
            "category": "myanmar_billing_payment"
        }
    ]
    
    for i, demo in enumerate(demo_responses, 1):
        print(f"\n{i}. Response Quality Analysis")
        print(f"   Query: {demo['query']}")
        print(f"   Response: {demo['response']}")
        
        analysis = analyzer.analyze_response_quality(
            demo['query'], 
            demo['response'], 
            demo['category']
        )
        
        print(f"   Quality Score: {analysis['quality_score']:.3f}")
        print(f"   Query Similarity: {analysis['query_similarity']['max_similarity']:.3f}")
    
    print("\n✅ Demo completed!")
    print("\nTo run full performance testing with semantic analysis:")
    print("python3 test_performance_enhanced.py --enable-semantics --save-report report.json")

if __name__ == "__main__":
    demo_semantic_analysis()
