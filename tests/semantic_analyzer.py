#!/usr/bin/env python3
"""
Semantic similarity analyzer for LLM performance testing.
Uses vector database to analyze semantic similarity between expected and actual responses.
"""

import numpy as np
import json
import pickle
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """Semantic similarity analyzer using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./semantic_cache"):
        """
        Initialize the semantic analyzer.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache embeddings and model
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load or initialize model
        self.model = self._load_model()
        
        # Storage for embeddings
        self.reference_embeddings = {}
        self.reference_texts = {}
        
    def _load_model(self) -> SentenceTransformer:
        """Load or download the sentence transformer model."""
        model_path = self.cache_dir / f"{self.model_name.replace('/', '_')}"
        
        if model_path.exists():
            logger.info(f"Loading cached model from {model_path}")
            return SentenceTransformer(str(model_path))
        else:
            logger.info(f"Downloading model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            model.save(str(model_path))
            return model
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def add_reference_texts(self, category: str, texts: List[str]) -> None:
        """
        Add reference texts for a category.
        
        Args:
            category: Category name (e.g., 'myanmar_internet_outage')
            texts: List of reference texts
        """
        logger.info(f"Adding {len(texts)} reference texts for category: {category}")
        embeddings = self.encode_texts(texts)
        
        self.reference_embeddings[category] = embeddings
        self.reference_texts[category] = texts
        
        # Save to cache
        self._save_category_cache(category, texts, embeddings)
    
    def _save_category_cache(self, category: str, texts: List[str], embeddings: np.ndarray) -> None:
        """Save category data to cache."""
        cache_file = self.cache_dir / f"{category}_cache.pkl"
        
        cache_data = {
            'texts': texts,
            'embeddings': embeddings,
            'model_name': self.model_name
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Cached {category} data to {cache_file}")
    
    def _load_category_cache(self, category: str) -> Optional[Tuple[List[str], np.ndarray]]:
        """Load category data from cache."""
        cache_file = self.cache_dir / f"{category}_cache.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if model matches
            if cache_data.get('model_name') != self.model_name:
                logger.warning(f"Model mismatch for {category}, regenerating...")
                return None
            
            return cache_data['texts'], cache_data['embeddings']
        except Exception as e:
            logger.warning(f"Failed to load cache for {category}: {e}")
            return None
    
    def load_reference_from_csv(self, csv_file: str) -> None:
        """
        Load reference texts from CSV file for semantic analysis.
        
        Args:
            csv_file: Path to CSV file with conversation pairs (reference/training data)
        """
        logger.info(f"Loading reference texts from {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Group by categories
        categories = {
            'myanmar_internet_outage': {'queries': [], 'responses': []},
            'myanmar_billing_payment': {'queries': [], 'responses': []},
            'myanmar_technical_support': {'queries': [], 'responses': []},
            'myanmar_product_inquiry': {'queries': [], 'responses': []},
            'myanmar_customer_service': {'queries': [], 'responses': []},
            'myanmar_mixed_language': {'queries': [], 'responses': []}
        }
        
        # Categorize based on content
        for _, row in df.iterrows():
            user_msg = str(row['user_message']).strip()
            response = str(row['response']).strip()
            
            if not user_msg or not response:
                continue
            
            # Simple categorization based on keywords
            category = self._categorize_text(user_msg)
            if category in categories:
                categories[category]['queries'].append(user_msg)
                categories[category]['responses'].append(response)
        
        # Add reference texts for each category
        for category, data in categories.items():
            if data['queries']:
                # Try to load from cache first
                cached_data = self._load_category_cache(category)
                if cached_data:
                    texts_cached, embeddings_cached = cached_data
                    self.reference_texts[category] = texts_cached
                    self.reference_embeddings[category] = embeddings_cached
                else:
                    self.add_reference_texts(category, data['queries'])
                
                # Also store responses for quality analysis
                response_category = f"{category}_responses"
                self.add_reference_texts(response_category, data['responses'])
        
        # Create combined category
        self._create_combined_category()
    
    def _create_combined_category(self) -> None:
        """Create a combined category from all Myanmar categories."""
        logger.info("Creating combined Myanmar category...")
        
        # Combine all Myanmar categories
        combined_texts = []
        for category in ['myanmar_internet_outage', 'myanmar_billing_payment', 
                        'myanmar_technical_support', 'myanmar_product_inquiry', 
                        'myanmar_customer_service', 'myanmar_mixed_language']:
            if category in self.reference_texts:
                combined_texts.extend(self.reference_texts[category])
        
        if combined_texts:
            # Remove duplicates while preserving order
            seen = set()
            unique_texts = []
            for text in combined_texts:
                if text not in seen:
                    seen.add(text)
                    unique_texts.append(text)
            
            logger.info(f"Combined category contains {len(unique_texts)} unique texts")
            self.add_reference_texts('myanmar_combined', unique_texts)
        else:
            logger.warning("No texts found to create combined category")
    
    def _categorize_text(self, text: str) -> str:
        """Simple text categorization based on keywords."""
        text_lower = text.lower()
        
        # Keywords for different categories
        if any(keyword in text for keyword in ['လိုင်းမကောင်း', 'လိုင်းကျ', 'အင်တာနက်မရ', 'လိုင်းပြတ်']):
            return 'myanmar_internet_outage'
        elif any(keyword in text for keyword in ['ဘေလ်ဆောင်', 'ငွေပေးဆောင်', 'ကြေး', 'ဘေဆောင်']):
            return 'myanmar_billing_payment'
        elif any(keyword in text for keyword in ['wifi', 'ချိတ်မရ', 'password', 'ပါစ်ဝါ']):
            return 'myanmar_technical_support'
        elif any(keyword in text for keyword in ['ပက်ကေ့ဂ်', 'အဝသုံး', 'ကဒ်', 'ပလန်']):
            return 'myanmar_product_inquiry'
        elif any(keyword in text for keyword in ['customer service', 'ဖုန်းနံပါတ်', 'ဆက်သွယ်']):
            return 'myanmar_customer_service'
        else:
            return 'myanmar_mixed_language'
    
    def calculate_similarity(self, query_text: str, category: str) -> Dict[str, float]:
        """
        Calculate semantic similarity between query and reference texts in a category.
        
        Args:
            query_text: Text to compare
            category: Category to compare against
            
        Returns:
            Dictionary with similarity scores and metadata
        """
        if category not in self.reference_embeddings:
            logger.warning(f"Category {category} not found in reference embeddings")
            logger.info(f"Available categories: {list(self.reference_embeddings.keys())}")
            return {'error': f'Category {category} not found'}
        
        # Encode query text
        query_embedding = self.encode_texts([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding, 
            self.reference_embeddings[category]
        )[0]
        
        # Get top similarities
        top_indices = np.argsort(similarities)[::-1][:5]  # Top 5
        
        result = {
            'query_text': query_text,
            'category': category,
            'max_similarity': float(np.max(similarities)),
            'mean_similarity': float(np.mean(similarities)),
            'top_similarities': [
                {
                    'text': self.reference_texts[category][idx],
                    'similarity': float(similarities[idx])
                }
                for idx in top_indices
            ]
        }
        
        return result
    
    def analyze_response_quality(self, query: str, response: str, expected_category: str) -> Dict[str, any]:
        """
        Analyze the quality of a response based on semantic similarity.
        
        Args:
            query: Original query
            response: Generated response
            expected_category: Expected category for the query
            
        Returns:
            Analysis results
        """
        # Calculate similarity for query
        query_analysis = self.calculate_similarity(query, expected_category)
        
        # Calculate similarity for response (if we have reference responses)
        response_analysis = None
        if f"{expected_category}_responses" in self.reference_embeddings:
            response_analysis = self.calculate_similarity(response, f"{expected_category}_responses")
        
        # Overall analysis
        analysis = {
            'query': query,
            'response': response,
            'expected_category': expected_category,
            'query_similarity': query_analysis,
            'response_similarity': response_analysis,
            'quality_score': self._calculate_quality_score(query_analysis, response_analysis)
        }
        
        return analysis
    
    def _calculate_quality_score(self, query_analysis: Dict, response_analysis: Optional[Dict]) -> float:
        """Calculate overall quality score."""
        if 'error' in query_analysis:
            return 0.0
        
        # Base score from query similarity
        base_score = query_analysis.get('max_similarity', 0.0)
        
        # Adjust based on response similarity if available
        if response_analysis and 'error' not in response_analysis:
            response_score = response_analysis.get('max_similarity', 0.0)
            # Weighted average (70% query, 30% response)
            base_score = 0.7 * base_score + 0.3 * response_score
        
        return base_score
    
    def batch_analyze(self, queries: List[str], responses: List[str], categories: List[str]) -> List[Dict]:
        """
        Analyze multiple query-response pairs.
        
        Args:
            queries: List of queries
            responses: List of responses
            categories: List of expected categories
            
        Returns:
            List of analysis results
        """
        if len(queries) != len(responses) or len(queries) != len(categories):
            raise ValueError("All input lists must have the same length")
        
        results = []
        for query, response, category in zip(queries, responses, categories):
            analysis = self.analyze_response_quality(query, response, category)
            results.append(analysis)
        
        return results
    
    def generate_report(self, analyses: List[Dict]) -> Dict[str, any]:
        """
        Generate a comprehensive report from multiple analyses.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            Comprehensive report
        """
        if not analyses:
            return {'error': 'No analyses provided'}
        
        # Calculate statistics
        quality_scores = [analysis['quality_score'] for analysis in analyses]
        
        # Category breakdown
        category_stats = {}
        for analysis in analyses:
            category = analysis['expected_category']
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(analysis['quality_score'])
        
        # Calculate category averages
        category_averages = {
            category: np.mean(scores) 
            for category, scores in category_stats.items()
        }
        
        report = {
            'total_analyses': len(analyses),
            'overall_quality_score': np.mean(quality_scores),
            'quality_score_std': np.std(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'category_averages': category_averages,
            'high_quality_count': sum(1 for score in quality_scores if score > 0.7),
            'medium_quality_count': sum(1 for score in quality_scores if 0.4 <= score <= 0.7),
            'low_quality_count': sum(1 for score in quality_scores if score < 0.4),
            'detailed_analyses': analyses
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str) -> None:
        """Save report to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {filename}")


def main():
    """Example usage of SemanticAnalyzer."""
    analyzer = SemanticAnalyzer()
    
    # Load reference texts from CSV
    analyzer.load_reference_from_csv('reference_conversions.csv')
    
    # Example analysis
    query = "လိုင်းမကောင်းပါဘူး ဘယ်လိုဖြစ်နေတာလဲဗျ"
    response = "အဆင်မပြေမှုများအတွက်တောင်းပန်ပါတယ်။ လူကြီးမင်း၏ Customer ID ကိုပေးပို့ပေးပါခင်ဗျာ။"
    
    analysis = analyzer.analyze_response_quality(query, response, 'myanmar_internet_outage')
    print("Analysis Result:")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
