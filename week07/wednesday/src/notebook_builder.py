import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("# Week 07 Wednesday · NLP Take-Home Assignment\n## Hard NLP Patterns & Aspect-Based Sentiment Analysis"))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from src.sentiment_patterns import analyze_patterns, get_baseline_failure_mode
from src.aspect_extractor import extract_aspect_sentiment, explain_aspect_hardness, get_improvement_strategies

# Load Wednesday Dataset
try:
    df = pd.read_csv('../data/ShopSense_Reviews_Wednesday.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Please run data_generator.py first.")

df.head()"""))

    cells.append(nbf.v4.new_markdown_cell("## Q1: Handling 5 Hardest NLP Patterns\nWe demonstrate how specific patterns (Negation, Sarcasm, Code-mixing, Implicit, Comparative) manifest and why baseline models typically fail."))

    cells.append(nbf.v4.new_code_cell("""# Filter reviews with specific patterns
pattern_reviews = df[df['pattern'] != 'Basic'].copy()

results = []
for idx, row in pattern_reviews.iterrows():
    text = row['review_text']
    detected = analyze_patterns(text)
    primary_pattern = detected[0] if detected else "None"
    failure = get_baseline_failure_mode(primary_pattern)
    
    results.append({
        "Review": text,
        "Pattern": primary_pattern,
        "Baseline Failure Mode": failure
    })

pd.DataFrame(results).style.set_properties(**{'text-align': 'left'})"""))

    cells.append(nbf.v4.new_markdown_cell("## Q2 (a) & (b): Aspect-Level Classification vs Review-Level"))

    cells.append(nbf.v4.new_code_cell("""print("--- Why is Aspect-Level Harder? ---")
for reason, detail in explain_aspect_hardness().items():
    print(f"- {reason}: {detail}")

print("\\n--- How to improve from 71% to 80%+? ---")
for i, strategy in enumerate(get_improvement_strategies()):
    print(f"{i+1}. {strategy}")"""))

    cells.append(nbf.v4.new_markdown_cell("## Q2 (c): Extracting Aspect-Sentiment Pairs\nDemonstrating how a single review can contain multiple conflicting sentiments aligned to different product traits."))

    cells.append(nbf.v4.new_code_cell("""test_rev = "Amazing camera quality but the battery is atrocious and customer support was unhelpful."
aspects = extract_aspect_sentiment(test_rev)

print(f"Original Review: '{test_rev}'")
print("\\nExtracted Aspects:")
for item in aspects:
    print(f" -> Aspect: {item['aspect']:<10} | Sentiment: {item['sentiment']}")"""))

    cells.append(nbf.v4.new_markdown_cell("### Conclusion:\nAs shown above, the review is simultaneously **Positive** (on Camera) and **Negative** (on Battery and Support). Review-level sentiment (e.g. 'Negative') would lose the critical praise for the hardware, which is why product teams prioritize aspect-level analysis."))

    nb['cells'] = cells
    out_dir = os.path.join(os.path.dirname(__file__), '../notebooks')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'week07_wednesday_assignment.ipynb')
    with open(out_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook saved to {out_path}")
    
if __name__ == '__main__':
    create_notebook()
