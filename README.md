# Enhanced vLLM Performance Testing with Semantic Analysis

This project provides enhanced performance testing for vLLM servers with semantic similarity analysis, specifically designed for Myanmar language conversation pairs.

## Features

- **Real Myanmar Conversation Data**: Uses actual conversation pairs from CSV files
- **Semantic Similarity Analysis**: Analyzes response quality using vector embeddings
- **Comprehensive Performance Metrics**: Response time, token generation, and semantic quality
- **Vector Database**: Caches embeddings for efficient similarity calculations
- **Detailed Reporting**: Generates comprehensive reports with recommendations

## Files Overview

### Core Files
- `semantic_analyzer.py` - Main semantic analysis engine
- `test_performance.py` - Performance testing script
- `test_fixtures.py` - Generated prompt variations for testing
- `demo_semantic.py` - Demo script for semantic analysis
- `generate_variations.py` - Script to generate prompt variations from reference data

### Data Files
- `reference_conversions.csv` - Reference Myanmar conversation pairs (training data for semantic analysis)
- `pyproject.toml` - Poetry configuration with dependencies

## Installation

### Using Poetry (Recommended)
```bash
# Install all dependencies including dev dependencies for semantic analysis
poetry install
```

### Using pip (Alternative)
```bash
# Install semantic analysis dependencies
bash install_deps.sh
# OR manually:
pip install sentence-transformers>=2.2.0 scikit-learn>=1.3.0 pandas>=1.5.0 numpy>=1.24.0 requests>=2.28.0
```

### Notes
- The semantic analyzer will automatically download and cache the sentence transformer model on first use
- Poetry manages dependencies in `pyproject.toml` under `[tool.poetry.group.dev.dependencies]`

## Usage

### Basic Performance Testing
```bash
python3 test_performance.py --server2-only
```

### With Expected vs Actual Response Comparison
```bash
python3 test_performance.py --server2-only --enable-semantics --save-report detailed_results.json
```

### Custom Token Limits (for longer responses)
```bash
python3 test_performance.py --server2-only --max-tokens 1000 --enable-semantics
```

### Test Specific Categories
```bash
python3 test_performance.py --prompt-category myanmar_internet_outage --enable-semantics
```

### Demo Semantic Analysis
```bash
python3 demo_semantic.py
```

## Semantic Analysis Features

### Categories Analyzed
- `myanmar_internet_outage` - Internet connectivity issues
- `myanmar_billing_payment` - Payment and billing inquiries  
- `myanmar_technical_support` - Technical support requests
- `myanmar_product_inquiry` - Product information requests
- `myanmar_customer_service` - General customer service
- `myanmar_mixed_language` - Mixed English/Myanmar conversations

### Quality Metrics
- **Semantic Similarity Score**: 0.0 to 1.0 (higher is better)
- **High Quality**: > 0.7 similarity
- **Medium Quality**: 0.4 - 0.7 similarity  
- **Low Quality**: < 0.4 similarity

### Performance Metrics
- Response time (average, median, min, max)
- Token generation rate
- Success rate
- Consistency (standard deviation)

## Configuration

### Server Configuration
The script tests two vLLM servers:
- **LLM1**: `aisingapore/Gemma-SEA-LION-v3-9B-IT` (Port 8000)
- **LLM2**: `NanEi/fr_sealion_merge_bot_v1-5` (Port 8001)

### Semantic Model
- Default: `all-MiniLM-L6-v2` (multilingual support)
- Cached in `./semantic_cache/` directory
- Automatically downloads on first use

## Example Output

```
🚀 Testing LLM2 (fr_sealion) (Port 8001)
Model: NanEi/fr_sealion_merge_bot_v1-5
Prompts: 50
--------------------------------------------------
  1. 1.234s | 45 tokens | 36.5 tok/s | Quality: 0.856
  2. 0.987s | 52 tokens | 52.7 tok/s | Quality: 0.743
  ...

📊 LLM2 (fr_sealion) Performance Summary:
============================================================
Total Requests:     50
Successful:         50
Failed:             0
Success Rate:       100.0%
Average Time:       1.123s
Median Time:        1.089s
Min Time:           0.856s
Max Time:           1.567s
Std Deviation:      0.234s

🎯 Semantic Quality Analysis:
Average Quality:    0.789
Median Quality:     0.801
Min Quality:        0.456
Max Quality:        0.923
Quality Std Dev:    0.156
High Quality (>0.7): 38
Medium Quality:     10
Low Quality (<0.4):  2
```

## Advanced Usage

### Custom Categories
You can add custom categories by modifying the `categorize_text()` method in `semantic_analyzer.py`.

### Batch Analysis
```python
from semantic_analyzer import SemanticAnalyzer

analyzer = SemanticAnalyzer()
analyzer.load_reference_from_csv('your_data.csv')

# Batch analyze multiple queries
queries = ["query1", "query2", "query3"]
responses = ["response1", "response2", "response3"] 
categories = ["category1", "category2", "category3"]

results = analyzer.batch_analyze(queries, responses, categories)
report = analyzer.generate_report(results)
```

### Custom Models
```python
# Use a different sentence transformer model
analyzer = SemanticAnalyzer(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Check internet connection and try again
2. **CSV Encoding Issues**: Ensure CSV is UTF-8 encoded
3. **Memory Issues**: Reduce batch size or use smaller model
4. **Server Connection**: Ensure vLLM servers are running on correct ports

### Performance Tips

- Use SSD storage for better caching performance
- Increase batch size for better GPU utilization
- Use smaller models for faster inference
- Cache embeddings to avoid recomputation

## Contributing

1. Add new conversation categories in `update_fixtures.py`
2. Extend semantic analysis metrics in `semantic_analyzer.py`
3. Add new performance metrics in `test_performance_enhanced.py`

## License

This project is open source. Please ensure compliance with the licenses of the underlying models and libraries used.