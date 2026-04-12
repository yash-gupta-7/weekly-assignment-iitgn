import re
from typing import Dict, List, Tuple

def detect_negation(text: str) -> bool:
    """Simple negation detector for 'not ... at all' type patterns."""
    pattern = r"not\s+.*?\s+at\s+all"
    return bool(re.search(pattern, text, re.IGNORECASE))

def detect_sarcasm(text: str) -> bool:
    """Simple sarcasm heuristic: Positive exclamation followed by immediate negative outcome."""
    positive_words = ["wow", "great", "amazing", "excellent"]
    negative_outcomes = ["broke", "failed", "died", "bad", "terrible"]
    
    words = text.lower().split()
    has_pos = any(pw in words for pw in positive_words)
    has_neg = any(nw in words for nw in negative_outcomes)
    
    return has_pos and has_neg

def detect_code_mixing(text: str) -> bool:
    """Heuristic for Hinglish (Hindi word presence)."""
    hindi_keywords = ["bahut", "accha", "lekin", "thi", "hai", "kam", "jyada"]
    words = text.lower().split()
    return any(hk in words for hk in hindi_keywords)

def detect_implicit_sentiment(text: str) -> bool:
    """Identifies implicit negative actions like 'returned', 'refunded'."""
    implicit_neg = ["returned", "refund", "waste", "throwing"]
    words = text.lower().split()
    return any(inw in words for inw in implicit_neg)

def detect_comparative(text: str) -> bool:
    """Identifies comparative structures like 'better than', 'superior to'."""
    pattern = r"better\s+than|superior\s+to|worse\s+than"
    return bool(re.search(pattern, text, re.IGNORECASE))

def analyze_patterns(text: str) -> List[str]:
    """Runs all detectors and returns found patterns."""
    found = []
    if detect_negation(text): found.append("Negation")
    if detect_sarcasm(text): found.append("Sarcasm")
    if detect_code_mixing(text): found.append("Code-mixing")
    if detect_implicit_sentiment(text): found.append("Implicit")
    if detect_comparative(text): found.append("Comparative")
    return found

def get_baseline_failure_mode(pattern: str) -> str:
    """Explains why a standard VADER or simple BOW might fail on this pattern."""
    failures = {
        "Negation": "Standard BOW sees 'not' as negative, missing the flip from 'at all' making it positive.",
        "Sarcasm": "Bag-of-words weights 'great' as positive, missing the logical contradiction of 'Broke'.",
        "Code-mixing": "Standard English tokenizers and lexicons (VADER) treat Hindi words as unknown/neutral (0 score).",
        "Implicit": "Sentiment lexicons often miss actions (verb phrases) like 'Returned' which carry massive negative signal.",
        "Comparative": "Models may score both brands (e.g. Samsung) without understanding the relative preference direction."
    }
    return failures.get(pattern, "General semantic gap.")
