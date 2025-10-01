# Test Fixtures for vLLM Performance Testing
# Contains various prompt categories for testing different models

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

# Myanmar language prompts for LLM2 (fr_sealion) - ISP Product Queries
myanmar_isp_product_prompts = [
    "အင်တာနက်ပက်ကေ့ဂ်တွေက ဘယ်လိုမျိုးတွေရှိလဲ?",
    "အင်တာနက်အမြန်နှုန်းက ဘယ်လောက်ရှိလဲ?",
    "အင်တာနက်ပက်ကေ့ဂ်တွေရဲ့ ဈေးနှုန်းက ဘယ်လောက်လဲ?",
    "အိမ်သုံးအင်တာနက်ပက်ကေ့ဂ်က ဘယ်လိုမျိုးတွေရှိလဲ?",
    "ကုမ္ပဏီသုံးအင်တာနက်ပက်ကေ့ဂ်က ဘယ်လိုမျိုးတွေရှိလဲ?",
    "အင်တာနက်ပက်ကေ့ဂ်တွေမှာ ဘယ်လိုအပိုဝန်ဆောင်မှုတွေပါလဲ?",
    "အင်တာနက်ပက်ကေ့ဂ်တွေကို ဘယ်လိုရွေးချယ်ရမလဲ?",
    "အင်တာနက်ပက်ကေ့ဂ်တွေရဲ့ အားသာချက်တွေက ဘာတွေလဲ?",
    "အင်တာနက်ပက်ကေ့ဂ်တွေကို ဘယ်လိုအပ်ချက်တွေအတွက် သုံးလို့ရလဲ?",
    "အင်တာနက်ပက်ကေ့ဂ်တွေရဲ့ ကန့်သတ်ချက်တွေက ဘာတွေလဲ?",
]

# Myanmar language prompts for LLM2 (fr_sealion) - Internet Outage Complaints
myanmar_outage_complaint_prompts = [
    "အင်တာနက်က မရဘူး၊ ဘာလုပ်ရမလဲ?",
    "အင်တာနက်အမြန်နှုန်းက နှေးနေတယ်၊ ဘာဖြစ်နေတာလဲ?",
    "အင်တာနက်က ခဏခဏပြတ်နေတယ်၊ ဘာလုပ်ရမလဲ?",
    "အင်တာနက်က မရဘူး၊ ဘယ်လိုဖြေရှင်းရမလဲ?",
    "အင်တာနက်အမြန်နှုန်းက နှေးနေတယ်၊ ဘာကြောင့်လဲ?",
    "အင်တာနက်က ခဏခဏပြတ်နေတယ်၊ ဘာကြောင့်လဲ?",
    "အင်တာနက်က မရဘူး၊ ဘယ်လိုဆက်သွယ်ရမလဲ?",
    "အင်တာနက်အမြန်နှုန်းက နှေးနေတယ်၊ ဘယ်လိုဖြေရှင်းရမလဲ?",
    "အင်တာနက်က ခဏခဏပြတ်နေတယ်၊ ဘယ်လိုဖြေရှင်းရမလဲ?",
    "အင်တာနက်က မရဘူး၊ ဘယ်လိုအကူအညီတောင်းရမလဲ?",
]

# Combined Myanmar prompts for LLM2
myanmar_combined_prompts = myanmar_isp_product_prompts + myanmar_outage_complaint_prompts

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
all_prompts = {
    "general": general_prompts,
    "myanmar_isp_products": myanmar_isp_product_prompts,
    "myanmar_outage_complaints": myanmar_outage_complaint_prompts,
    "myanmar_combined": myanmar_combined_prompts,
    "technical": technical_prompts,
    "creative": creative_prompts,
    "customer_service": customer_service_prompts,
}
