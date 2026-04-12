from typing import List, Dict, Tuple

def extract_aspect_sentiment(text: str) -> List[Dict[str, str]]:
    """
    Extracts aspects and their associated sentiments from a review string.
    This is a rule-based implementation for demonstration purposes.
    """
    aspects_map = {
        "camera": ["amazing", "great", "excellent", "stunning"],
        "battery": ["atrocious", "bad", "terrible", "drains"],
        "support": ["unhelpful", "slow", "rude"],
        "fabric": ["soft", "comfortable"],
        "stitching": ["loose", "poor"],
        "food": ["delicious", "tasty"],
        "service": ["slow", "bad"]
    }
    
    results = []
    text_lower = text.lower()
    
    # Split by common contrastive conjunctions to help isolate sentiments
    parts = []
    if " but " in text_lower:
        parts = text_lower.split(" but ")
    elif " however " in text_lower:
        parts = text_lower.split(" however ")
    else:
        parts = [text_lower]
        
    for part in parts:
        for aspect, keywords in aspects_map.items():
            if aspect in part:
                # Find which keyword is present
                sentiment = "Neutral"
                # Check for positive keywords
                if any(kw in part for kw in ["amazing", "great", "excellent", "soft", "delicious"]):
                    sentiment = "Positive"
                elif any(kw in part for kw in ["atrocious", "unhelpful", "bad", "terrible", "slow", "loose"]):
                    sentiment = "Negative"
                
                results.append({"aspect": aspect, "sentiment": sentiment})
                
    return results

def explain_aspect_hardness() -> Dict[str, str]:
    """Provides reasons why aspect-level classification is harder than review-level."""
    return {
        "Data Sparsity": "Aspect-level requires labeling every entity, not just the whole doc.",
        "Sentiment Ambiguity": "The same word (e.g., 'hot') can be positive for food but negative for a phone.",
        "Coreference Resolution": "Identifying that 'it' refers to the 'camera' and not the 'case' in the next sentence.",
        "Overlapping Sentiments": "A single sentence can praise one aspect while criticizing another."
    }

def get_improvement_strategies() -> List[str]:
    """Strategies to improve aspect-level F1 scores."""
    return [
        "Dependency Parsing: Linking adjectives to specifically modified nouns.",
        "Domain-Specific Embeddings: Training on e-commerce data to capture product-specific jargon.",
        "Transfer Learning: Using BERT/RoBERTa focused on Aspect-Based Sentiment Analysis (ABSA).",
        "Data Augmentation: Synthesizing more balanced examples for rare aspects."
    ]
