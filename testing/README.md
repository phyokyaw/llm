# Testing Directory

This directory contains all testing-related Python files for the vLLM performance testing and semantic analysis system.

## Files

- **`test_performance.py`** - Main performance testing script with semantic analysis
- **`test_fixtures.py`** - Test fixtures containing various prompt categories
- **`demo_semantic.py`** - Demo script for semantic similarity analysis
- **`quick_test.py`** - Quick performance test script
- **`generate_variations.py`** - Script to generate prompt variations from reference conversations
- **`__init__.py`** - Package initialization file

## Usage

### Running Performance Tests
```bash
# From the project root directory
python3 -m testing.test_performance --enable-semantics
```

### Running Quick Tests
```bash
# From the project root directory
python3 -m testing.quick_test
```

### Running Semantic Analysis Demo
```bash
# From the project root directory
python3 -m testing.demo_semantic
```

### Generating Prompt Variations
```bash
# From the project root directory
python3 -m testing.generate_variations
```

## Dependencies

The testing scripts depend on:
- `reference_conversions.csv` (in this directory)
- Various Python packages: requests, pandas, sentence-transformers, etc.

## Import Structure

The testing package can be imported as:
```python
from testing import all_prompts, EnhancedPerformanceTester, demo_semantic_analysis, SemanticAnalyzer
```
