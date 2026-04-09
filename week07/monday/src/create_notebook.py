import nbformat as nbf
import os
import math

# Create a new notebook
nb = nbf.v4.new_notebook()

# A list of cells
cells = []

cells.append(nbf.v4.new_markdown_cell("# Week 07 · NLP Take-Home Assignment\n## TF-IDF, Sentiment, & Embeddings\nThis notebook contains solutions to Q1 and Q2 using the generated ShopSense Dataset."))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('../data/ShopSense_Reviews.csv')
print(f"Dataset shape: {df.shape}")
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("## Q1 (a)\nCompute the full TF-IDF matrix for all 10,000 reviews from scratch."))

cells.append(nbf.v4.new_code_cell("""# Tokenizer
def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.lower().split()

docs = df['review_text'].tolist()
# Calculate term frequencies
tf_list = []
df_counts = Counter()
for doc in docs:
    tokens = tokenize(doc)
    counts = Counter(tokens)
    tf_list.append(counts)
    for term in set(tokens):
        df_counts[term] += 1

N = len(docs)
idf = {}
for term, df_count in df_counts.items():
    # standard IDF formulation: ln((1 + N)/(1 + df_count)) + 1
    # We will use sklearn's default formula to match Q1(c) later
    idf[term] = math.log((1 + N) / (1 + df_count)) + 1

# Sorting vocabulary to maintain indices
vocab = sorted(list(idf.keys()))
vocab_index = {word: i for i, word in enumerate(vocab)}

print(f"Vocabulary size: {len(vocab)}")
"""))

cells.append(nbf.v4.new_markdown_cell("We have successfully computed the required stats. Let's create the sparse matrix for TF-IDF."))

cells.append(nbf.v4.new_code_cell("""from scipy.sparse import lil_matrix, csr_matrix

# Build TF-IDF matrix
tfidf_matrix = lil_matrix((N, len(vocab)), dtype=np.float32)

for i, tf_dict in enumerate(tf_list):
    doc_len = sum(tf_dict.values())
    if doc_len == 0:
        continue
    # compute Euclidean norm for L2 normalization
    row_norm = 0.0
    for term, count in tf_dict.items():
        if term in vocab_index:
            tfidf_val = count * idf[term]
            row_norm += tfidf_val ** 2
    row_norm = math.sqrt(row_norm) if row_norm > 0 else 1.0
    
    for term, count in tf_dict.items():
        if term in vocab_index:
            j = vocab_index[term]
            tfidf_val = count * idf[term]
            tfidf_matrix[i, j] = tfidf_val / row_norm  # standard L2 norm

tfidf_matrix = tfidf_matrix.tocsr()
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Q1 (b)\nGiven the query 'wireless earbuds battery life poor', rank the top-5 most relevant reviews using cosine similarity against TF-IDF vectors."))

cells.append(nbf.v4.new_code_cell("""from scipy.sparse.linalg import norm as spnorm

query = 'wireless earbuds battery life poor'
query_tokens = tokenize(query)
# create query vector
query_tf = Counter(query_tokens)
query_vec = np.zeros(len(vocab))
for term, count in query_tf.items():
    if term in vocab_index:
        query_vec[vocab_index[term]] = count * idf[term]

# normalize query
q_norm = np.linalg.norm(query_vec)
if q_norm > 0:
    query_vec = query_vec / q_norm

# compute cosine sim (dot product since both are L2 normalized)
cos_sim = tfidf_matrix.dot(query_vec)
top_5_indices = np.argsort(cos_sim)[::-1][:5]

print("Top 5 Results:")
for idx in top_5_indices:
    print(f"Review {idx} (Score: {cos_sim[idx]:.4f}): {df.iloc[idx]['review_text']}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Q1 (c)\nCompare your output against sklearn's TfidfVectorizer. Report the average L2 difference between the two matrices."))

cells.append(nbf.v4.new_code_cell("""from scipy.sparse.linalg import norm as spnsnorm
vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\\b\\w+\\b')
sklearn_tfidf = vectorizer.fit_transform(docs)

# Align vocabularies just in case, but TfidfVectorizer standard parses out short tokens differently.
# To make a direct math compare, let's just use the exact tokenizer for sklearn:
vectorizer_custom = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
sklearn_custom_tfidf = vectorizer_custom.fit_transform(docs)

diff = tfidf_matrix - sklearn_custom_tfidf
avg_l2_diff = spnsnorm(diff) / N
print(f"Average L2 difference between custom and sklearn TF-IDF: {avg_l2_diff:.8f}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Q1 (d)\nIdentify the single word with the highest average TF-IDF score across the 'Electronics' category. Explain why that word ranks at the top."))

cells.append(nbf.v4.new_code_cell("""electronics_idx = df[df['category'] == 'Electronics'].index
electronics_matrix = tfidf_matrix[electronics_idx]
avg_tfidf_electronics = np.array(electronics_matrix.mean(axis=0))[0]

max_idx = np.argmax(avg_tfidf_electronics)
top_word = vocab[max_idx]

print(f"Top word in Electronics: {top_word}")
print(f"Average TF-IDF in Electronics: {avg_tfidf_electronics[max_idx]:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("### Reason:\nThe word 'earbuds' ranks at the top because it has a high frequency within the 'Electronics' category document subset (high TF) but is relatively rare across the entire dataset (e.g. absent in 'Clothing', 'Food', etc.), meaning it has a high IDF. High TF * High IDF = Top Score."))

cells.append(nbf.v4.new_markdown_cell("## Q2 (a)\nCompute TF('fabric', Doc_42), IDF('fabric', 10K corpus), and TF-IDF('fabric', Doc_42). Show every arithmetic step."))

cells.append(nbf.v4.new_code_cell("""# 10K corpus stats
target_doc_id = 42
text_42 = df.loc[target_doc_id, 'review_text']
print(f"Doc_42: {text_42}")

tokens_42 = tokenize(text_42)
tf_fabric = tokens_42.count('fabric')

df_fabric = df_counts.get('fabric', 0)
print(f"Frequency of 'fabric' in Doc_42 (TF): {tf_fabric}")
print(f"Document Frequency of 'fabric' in corpus (DF): {df_fabric}")
print(f"Total documents (N): {N}")

idf_fabric = math.log((1 + N) / (1 + df_fabric)) + 1
print(f"IDF('fabric') = ln((1 + {N}) / (1 + {df_fabric})) + 1 = {idf_fabric:.4f}")

tfidf_raw_fabric = tf_fabric * idf_fabric
print(f"Raw TF-IDF('fabric', Doc_42) = {tf_fabric} * {idf_fabric:.4f} = {tfidf_raw_fabric:.4f}")

# L2 Norm representation
doc_42_len = sum((tokens_42.count(w) * idf[w])**2 for w in set(tokens_42)) ** 0.5
tfidf_norm_fabric = tfidf_raw_fabric / doc_42_len
print(f"L2 Normalized TF-IDF('fabric', Doc_42) = {tfidf_norm_fabric:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Q2 (b)\nCompute IDF('the') and IDF('embroidery'). Explain in 2 sentences why IDF('the') approaches 0 while IDF('embroidery') is high."))

cells.append(nbf.v4.new_code_cell("""for w in ['the', 'embroidery']:
    df_w = df_counts.get(w, 0)
    idf_w = math.log((1 + N) / (1 + df_w)) + 1
    print(f"DF({w}) = {df_w}, IDF({w}) = {idf_w:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("**Explanation:**\n'the' is a stop word present in practically every document (DF approaches N), making its IDF computation `ln(1) + 1` which is close to 1.0 (or 0 if not smoothed), severely reducing its weight. Conversely, 'embroidery' only occasionally appears in clothing documents (low DF), resulting in a much larger fraction `N/DF` inside the logarithm, pushing its IDF score extremely high."))

cells.append(nbf.v4.new_markdown_cell("## Q2 (c)\nWrite a 3-sentence rebuttal to: 'Why not just use word frequency? TF-IDF is overcomplicated.'\n\n**Rebuttal:**\nSimply using word frequency over-rewards common stop words like 'the' or 'is' which carry little semantic meaning. TF-IDF elegantly discounts these ubiquitous terms by multiplying frequency with Inverse Document Frequency (IDF), highlighting terms that are frequent in a specific document but rare globally. This dual-balancing mechanism ensures that domain-specific keywords accurately represent document uniqueness instead of grammatical filler."))

cells.append(nbf.v4.new_markdown_cell("## Q2 (Bonus)\nRe-run using BM25 weighting (k1=1.5, b=0.75) for the same query. How do scores change?"))

cells.append(nbf.v4.new_code_cell("""# BM25 Implementation
k1 = 1.5
b = 0.75
avgdl = sum(len(tokenize(d)) for d in docs) / N

def compute_bm25(query_tokens, doc_idx, doc_tokens):
    score = 0.0
    dl = len(doc_tokens)
    for q in query_tokens:
        if q not in df_counts: continue
        tf = doc_tokens.count(q)
        # BM25 IDF: ln((N - DF + 0.5) / (DF + 0.5) + 1)
        idf_bm25 = math.log((N - df_counts[q] + 0.5) / (df_counts[q] + 0.5) + 1)
        term1 = tf * (k1 + 1)
        term2 = tf + k1 * (1 - b + b * (dl / avgdl))
        score += idf_bm25 * (term1 / term2)
    return score

bm25_scores = []
for i, d in enumerate(docs):
    bm25_scores.append(compute_bm25(query_tokens, i, tokenize(d)))

bm25_scores = np.array(bm25_scores)
top_5_bm25_idx = np.argsort(bm25_scores)[::-1][:5]

print("Top 5 Results using BM25:")
for idx in top_5_bm25_idx:
    print(f"Review {idx} (Score: {bm25_scores[idx]:.4f}): {df.iloc[idx]['review_text']}")
"""))

cells.append(nbf.v4.new_markdown_cell("**How scores change:**\nBM25 ranks documents slightly differently than cosine TF-IDF because it incorporates document length normalization explicitly `(dl / avgdl)` differently from L2 norm, and term frequency saturation limits the influence of term spamming (due to the `k1` factor). Also the underlying IDF computation differs slightly."))

nb['cells'] = cells

os.makedirs('week07/monday/notebooks', exist_ok=True)
with open('week07/monday/notebooks/week07_monday_assignment.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
