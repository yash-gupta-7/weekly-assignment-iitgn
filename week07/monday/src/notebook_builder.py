import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("# Week 07 · NLP Take-Home Assignment\n## TF-IDF, Sentiment, & Embeddings\nThis notebook demonstrates solutions to Q1 and Q2 using modular python architecture."))

    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from src.data_generator import create_shopsense_reviews
from src.tfidf_engine import compute_corpus_stats, build_tfidf_matrix, compute_cosine_similarity
from src.analytical_models import compute_manual_tfidf, compute_bm25_scores
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import norm as spnsnorm

# Generated if not exists
try:
    df = pd.read_csv('../data/ShopSense_Reviews.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Generating...")
    df = create_shopsense_reviews('../data/ShopSense_Reviews.csv')

print(f"Dataset shape: {df.shape}")
df.head(3)"""))

    cells.append(nbf.v4.new_markdown_cell("## Q1 (a): TF-IDF Matrix from Scratch"))

    cells.append(nbf.v4.new_code_cell("""docs = df['review_text'].fillna('').tolist()
tf_list, df_counts, idf = compute_corpus_stats(docs)

vocab = sorted(list(idf.keys()))
vocab_index = {word: i for i, word in enumerate(vocab)}

tfidf_matrix = build_tfidf_matrix(tf_list, idf, vocab)
print(f"Vocabulary size: {len(vocab)}")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")"""))

    cells.append(nbf.v4.new_markdown_cell("## Q1 (b): Ranking Query with Cosine Similarity"))

    cells.append(nbf.v4.new_code_cell("""query = 'wireless earbuds battery life poor'
cos_sim = compute_cosine_similarity(query, tfidf_matrix, idf, vocab_index)

top_5_indices = np.argsort(cos_sim)[::-1][:5]
print("Top 5 Results for Cosine:")
for idx in top_5_indices:
    print(f"[{idx}] Score: {cos_sim[idx]:.4f} | {df.iloc[idx]['review_text']}")"""))

    cells.append(nbf.v4.new_markdown_cell("## Q1 (c): Sklearn Comparison"))

    cells.append(nbf.v4.new_code_cell("""vectorizer_custom = TfidfVectorizer(tokenizer=lambda x: x.lower().split(), token_pattern=None, lowercase=False)
sklearn_custom_tfidf = vectorizer_custom.fit_transform(docs)

diff = tfidf_matrix - sklearn_custom_tfidf
avg_l2_diff = spnsnorm(diff) / len(docs)
print(f"Average L2 difference between custom and sklearn TF-IDF: {avg_l2_diff:.8f}")"""))

    cells.append(nbf.v4.new_markdown_cell("## Q1 (d): Electronics Category TF-IDF"))

    cells.append(nbf.v4.new_code_cell("""electronics_idx = df[df['category'] == 'Electronics'].index
electronics_matrix = tfidf_matrix[electronics_idx]
avg_tfidf_electronics = np.array(electronics_matrix.mean(axis=0))[0]

max_idx = np.argmax(avg_tfidf_electronics)
print(f"Top word in Electronics: {vocab[max_idx]}\nAverage Output TF-IDF: {avg_tfidf_electronics[max_idx]:.4f}")"""))
    cells.append(nbf.v4.new_markdown_cell("### Reason:\nThe word 'earbuds' ranks at the top because it has high term frequency in Electronics, but a very high global IDF since it doesn't appear in Clothing, Food, etc."))

    cells.append(nbf.v4.new_markdown_cell("## Q2 (a): Manual TF-IDF for 'fabric'"))

    cells.append(nbf.v4.new_code_cell("""doc_42_text = df.loc[42, 'review_text']
print(f"Doc_42: {doc_42_text}")

tokens_42 = doc_42_text.lower().split()
stats = compute_manual_tfidf('fabric', tokens_42, len(docs), df_counts.get('fabric', 0))

print(f"TF: {stats['tf']}")
print(f"IDF: {stats['idf']:.4f}")
print(f"Raw TF-IDF ('fabric'): {stats['tfidf_raw']:.4f}")"""))

    cells.append(nbf.v4.new_markdown_cell("## Q2 (b): IDF('the') vs IDF('embroidery')"))

    cells.append(nbf.v4.new_code_cell("""for w in ['the', 'embroidery']:
    stat = compute_manual_tfidf(w, [], len(docs), df_counts.get(w, 0))
    print(f"IDF({w}) = {stat['idf']:.4f}")"""))
    
    cells.append(nbf.v4.new_markdown_cell("**Explanation:**\n'the' is a stop word appearing in almost every document, pushing DF towards N and making `ln(1)` = 0. 'embroidery' is rare, making `DF` tiny, giving it a massive IDF weight."))

    cells.append(nbf.v4.new_markdown_cell("## Q2 (c): Rebuttal\nSimply using word frequency over-rewards common stop words like 'the' or 'is' which carry little semantic meaning. TF-IDF elegantly discounts these ubiquitous terms by multiplying frequency with Inverse Document Frequency (IDF), highlighting terms that are frequent in a specific document but rare globally. This dual-balancing mechanism ensures that domain-specific keywords accurately represent document uniqueness instead of grammatical filler."))

    cells.append(nbf.v4.new_markdown_cell("## Q2 (Bonus): BM25 Variant"))

    cells.append(nbf.v4.new_code_cell("""bm25_scores = compute_bm25_scores(docs, query.split(), df_counts, len(docs))
top_5_bm25_idx = np.argsort(bm25_scores)[::-1][:5]

print("Top 5 Results using BM25:")
for idx in top_5_bm25_idx:
    print(f"[{idx}] Score: {bm25_scores[idx]:.4f} | {df.iloc[idx]['review_text']}")"""))

    nb['cells'] = cells
    os.makedirs(os.path.join(os.path.dirname(__file__), '../notebooks'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '../notebooks/week07_monday_assignment.ipynb')
    with open(out_path, 'w') as f:
        nbf.write(nb, f)
    
if __name__ == '__main__':
    create_notebook()
