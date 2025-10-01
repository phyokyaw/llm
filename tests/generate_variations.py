#!/usr/bin/env python3
"""
Generate prompt variations for testing based on reference conversations.
Creates variations of user prompts while keeping the reference data for semantic analysis.
"""

import csv
import random
import re
from typing import List, Dict, Tuple

def generate_prompt_variations(reference_file: str) -> Dict[str, List[str]]:
    """
    Generate variations of prompts based on reference conversations.
    
    Args:
        reference_file: Path to reference conversations CSV
        
    Returns:
        Dictionary of categorized prompt variations
    """
    
    # Load reference data
    reference_data = []
    with open(reference_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            reference_data.append({
                'user_message': row['user_message'].strip(),
                'response': row['response'].strip()
            })
    
    # Generate variations
    variations = {
        'myanmar_internet_outage': [],
        'myanmar_billing_payment': [],
        'myanmar_technical_support': [],
        'myanmar_product_inquiry': [],
        'myanmar_customer_service': [],
        'myanmar_mixed_language': []
    }
    
    for data in reference_data:
        user_msg = data['user_message']
        category = categorize_conversation(user_msg, data['response'])
        
        # Generate variations for each original prompt
        prompt_variations = create_variations(user_msg)
        
        # Add to appropriate category
        if category in variations:
            variations[category].extend(prompt_variations)
    
    # Remove duplicates while preserving order
    for category in variations:
        seen = set()
        unique_variations = []
        for variation in variations[category]:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)
        variations[category] = unique_variations
    
    return variations

def categorize_conversation(user_msg: str, response: str) -> str:
    """Categorize conversation based on content analysis."""
    
    # Keywords for different categories
    outage_keywords = ['လိုင်းမကောင်း', 'လိုင်းကျ', 'အင်တာနက်မရ', 'လိုင်းပြတ်', 'လိုင်းနှေး', 'မရဘူး', 'ပြတ်နေ', 'လိုင်းလုံးဝမကောင်း']
    billing_keywords = ['ဘေလ်ဆောင်', 'ငွေပေးဆောင်', 'ကြေး', 'ဘေဆောင်', 'ငွေဖြည့်', 'ပေးဆောင်', 'ဘေလ်ပေးဆောင်', 'ငွေချေ']
    technical_keywords = ['wifi', 'Wi-Fi', 'ချိတ်မရ', 'ချိတ်ဆက်', 'password', 'ပါစ်ဝါ', 'ကဒ်ပျက်', 'ခဲခြစ်', 'ပျက်သွား']
    product_keywords = ['ပက်ကေ့ဂ်', 'အဝသုံး', 'ကဒ်', 'ပလန်', 'ဈေးနှုန်း', 'အမြန်နှုန်း', 'Mbps', 'ကတ်တွေ', 'ပလန်တွေ']
    service_keywords = ['customer service', 'ဖုန်းနံပါတ်', 'ဆက်သွယ်', 'အကူအညီ', 'ပြောချင်', 'ဖုန်းနံပါတ်သိချင်']
    
    # Check for mixed language (English + Myanmar)
    has_english = bool(re.search(r'[a-zA-Z]', user_msg))
    has_myanmar = bool(re.search(r'[\u1000-\u109F]', user_msg))
    
    if has_english and has_myanmar:
        return 'myanmar_mixed_language'
    
    # Check categories
    if any(keyword in user_msg for keyword in outage_keywords):
        return 'myanmar_internet_outage'
    elif any(keyword in user_msg for keyword in billing_keywords):
        return 'myanmar_billing_payment'
    elif any(keyword in user_msg for keyword in technical_keywords):
        return 'myanmar_technical_support'
    elif any(keyword in user_msg for keyword in product_keywords):
        return 'myanmar_product_inquiry'
    elif any(keyword in user_msg for keyword in service_keywords):
        return 'myanmar_customer_service'
    else:
        return 'myanmar_mixed_language'

def create_variations(original_prompt: str) -> List[str]:
    """
    Create variations of a prompt while maintaining semantic meaning.
    
    Args:
        original_prompt: Original prompt text
        
    Returns:
        List of prompt variations
    """
    variations = [original_prompt]  # Include original
    
    # Variation patterns for Myanmar text
    variation_patterns = [
        # Add politeness markers
        lambda text: f"မင်္ဂလာပါ {text}",
        lambda text: f"{text} ခင်ဗျာ",
        lambda text: f"{text} ရှင့်",
        
        # Add urgency indicators
        lambda text: f"အရေးကြီးပါတယ် {text}",
        lambda text: f"အမြန်ဆုံး {text}",
        lambda text: f"အရေးတကြီး {text}",
        
        # Add context
        lambda text: f"အိမ်မှာ {text}",
        lambda text: f"ရုံးမှာ {text}",
        lambda text: f"အလုပ်မှာ {text}",
        
        # Simplify complex sentences
        lambda text: simplify_sentence(text),
        
        # Add emotional context
        lambda text: f"စိတ်ရှုပ်နေပါတယ် {text}",
        lambda text: f"အဆင်မပြေတာ {text}",
    ]
    
    # Apply variations
    for pattern in variation_patterns:
        try:
            variation = pattern(original_prompt)
            if variation != original_prompt and len(variation) < 200:  # Keep reasonable length
                variations.append(variation)
        except:
            continue
    
    # Add some English variations for mixed language prompts
    if re.search(r'[a-zA-Z]', original_prompt):
        english_variations = [
            lambda text: text.replace("I want to", "I need to"),
            lambda text: text.replace("How to", "Can you help me"),
            lambda text: text.replace("?", " please?"),
            lambda text: f"Hello, {text}",
            lambda text: f"Hi, {text}",
        ]
        
        for pattern in english_variations:
            try:
                variation = pattern(original_prompt)
                if variation != original_prompt:
                    variations.append(variation)
            except:
                continue
    
    # Limit to reasonable number of variations
    return variations[:8]  # Max 8 variations per original prompt

def simplify_sentence(text: str) -> str:
    """Simplify complex Myanmar sentences."""
    # Remove some complex parts while keeping core meaning
    simplifications = [
        (r'အဆင်မပြေမှုများအတွက်တောင်းပန်ပါတယ်။', ''),
        (r'လူကြီးမင်း၏', ''),
        (r'ပေးပို့ပြီးသားဖြစ်ပါကထပ်မံပေးပို့ရန်မလိုပါခင်ဗျာ။', ''),
        (r'Customer service agentဆီသို့လွှဲပေးပါမယ်ခင်ဗျာ။', ''),
        (r'Message များကို အလှည့်ကျပြန်လည်ဖြေကြားပေးမှာဖြစ်တဲ့အတွက် အချိန်ခဏလောက်စောင့်ဆိုင်းပေးရန် မေတ္တာရပ်ခံအပ်ပါတယ်။', ''),
    ]
    
    simplified = text
    for pattern, replacement in simplifications:
        simplified = re.sub(pattern, replacement, simplified)
    
    # Clean up extra spaces
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    
    return simplified if simplified != text else text

def generate_fixtures_code(variations: Dict[str, List[str]]) -> str:
    """Generate the updated fixtures code with variations."""
    
    code = '''# Test Fixtures for vLLM Performance Testing
# Contains various prompt categories with variations for testing different models

# General English prompts for LLM1 (Gemma)
general_prompts = [
    "Hello! How are you today?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are the benefits of renewable energy?",
    "Tell me a joke.",
    "How do you make a sandwich?",
    "What is machine learning?",
    "Describe the weather today.",
    "What is your favorite color and why?",
    "Explain photosynthesis briefly.",
    "What is the meaning of life?",
    "How do computers work?",
    "Tell me about space exploration.",
    "What is artificial intelligence?",
]

# Myanmar language prompt variations for testing
# Generated from reference conversations with semantic variations
'''
    
    # Add each category with variations
    for category, prompts in variations.items():
        if prompts:
            code += f'\n# {category.replace("_", " ").title()} prompts\n'
            code += f'{category} = [\n'
            for prompt in prompts:  # Use all prompts
                code += f'    "{prompt}",\n'
            code += ']\n'
    
    # Add combined category
    all_myanmar_prompts = []
    for category, prompts in variations.items():
        if category.startswith('myanmar_'):
            all_myanmar_prompts.extend(prompts)
    
    code += f'''
# Combined Myanmar prompts for LLM2
myanmar_combined_prompts = {all_myanmar_prompts}

# Technical prompts for both models
technical_prompts = [
    "What is the difference between HTTP and HTTPS?",
    "Explain how DNS works.",
    "What is a VPN and how does it work?",
    "Describe the OSI model layers.",
    "What is the difference between TCP and UDP?",
    "How does SSL/TLS encryption work?",
    "What is load balancing?",
    "Explain microservices architecture.",
    "What is containerization?",
    "How does caching improve performance?",
]

# Creative prompts for both models
creative_prompts = [
    "Write a short story about a robot.",
    "Create a poem about technology.",
    "Describe a futuristic city.",
    "Write a dialogue between two AI systems.",
    "Create a haiku about the internet.",
    "Describe a day in the life of a programmer.",
    "Write a limerick about cybersecurity.",
    "Create a short script for a tech comedy.",
    "Describe an alien's first encounter with Earth's internet.",
    "Write a song about digital transformation.",
]

# Customer service prompts
customer_service_prompts = [
    "How can I reset my password?",
    "I forgot my username, what should I do?",
    "How do I contact customer support?",
    "What are your business hours?",
    "How can I cancel my subscription?",
    "I want to upgrade my plan, how do I do that?",
    "How do I get a refund?",
    "What payment methods do you accept?",
    "How do I update my billing information?",
    "I'm having trouble logging in, can you help?",
]

# All prompts combined
all_prompts = {{
    "general": general_prompts,
'''
    
    # Add Myanmar categories
    for category, prompts in variations.items():
        if category.startswith('myanmar_'):
            code += f'    "{category}": {category},\n'
    
    code += '''    "myanmar_combined": myanmar_combined_prompts,
    "technical": technical_prompts,
    "creative": creative_prompts,
    "customer_service": customer_service_prompts,
}
'''
    
    return code

def main():
    """Generate prompt variations and update fixtures."""
    print("🔄 Generating prompt variations from reference conversations...")
    
    # Generate variations
    import os
    csv_path = os.path.join(os.path.dirname(__file__),
                            'reference_conversions.csv')
    variations = generate_prompt_variations(csv_path)
    
    # Print statistics
    print("\n📊 Generated Prompt Variations:")
    total_variations = 0
    for category, prompts in variations.items():
        if prompts:
            print(f"  {category}: {len(prompts)} variations")
            total_variations += len(prompts)
    
    print(f"\n📝 Total variations generated: {total_variations}")
    
    # Generate fixtures code
    print("\n✍️ Generating updated fixtures...")
    fixtures_code = generate_fixtures_code(variations)
    
    # Write to file
    import os
    fixtures_path = os.path.join(os.path.dirname(__file__), 'test_fixtures.py')
    with open(fixtures_path, 'w', encoding='utf-8') as f:
        f.write(fixtures_code)
    
    print("✅ Updated test_fixtures.py with prompt variations!")
    print("\n💡 Usage:")
    print("  - Reference conversations are used for semantic analysis")
    print("  - Prompt variations are used for testing LLM responses")
    print("  - Run: python3 -m tests.demo_semantic to test semantic analysis")

if __name__ == "__main__":
    main()


