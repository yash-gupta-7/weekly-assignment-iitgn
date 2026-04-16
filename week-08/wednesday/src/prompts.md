# AI Prompts & Critique Log — Week 08 Wednesday

## Sub-step 3: CNN Architecture
**Prompt used:**
> "Generate a modular 2-layer CNN in PyTorch for MNIST digit classification. It should be in a separate class with a forward() method. Include a get_device() utility."

**AI Output Assessment:**
- ✅ Correct architecture with Conv2d layers and MaxPool2d.
- ❌ AI did not include MPS support for Apple Silicon. Added `torch.backends.mps.is_available()` manually.
- ❌ AI used a single flat cell — refactored into `model_utils.py` for modularity.

---

## Sub-step 4: Hate Speech Classifier
**Prompt used:**
> "Build a hate speech classifier using TF-IDF and Logistic Regression in scikit-learn on an imbalanced binary dataset. Show the classification report."

**AI Output Assessment:**
- ✅ Correct use of `TfidfVectorizer` and `LogisticRegression`.
- ❌ AI did NOT add `class_weight='balanced'` — critical for handling imbalance. Added manually.
- ❌ AI did not include train/test stratification. Added `stratify=y` to the split.

---

## Sub-step 4: Semantic Search
**Prompt used:**
> "Write a semantic search function using sentence-transformers and cosine similarity to find top-k most similar posts from a list of texts."

**AI Output Assessment:**
- ✅ Correct use of `SentenceTransformer.encode()` and `util.cos_sim()`.
- ❌ AI returned itself as the top hit (query post was index 0 of itself). Added logic to skip index 0 in results.

---

## Sub-step 6: TF-IDF vs S-BERT Comparison
**Prompt used:**
> "Compare TF-IDF cosine similarity with Sentence-BERT embeddings for semantic search. Show side-by-side results for the same query."

**AI Output Assessment:**
- ✅ Structurally correct comparison approach.
- ❌ AI did not connect the CNN filter analogy. Added the explanation linking learned CNN filters (local patterns → global concepts) to how S-BERT captures semantic context that TF-IDF misses.

---

## Sub-step 7: Transfer Learning Analysis
**Prompt used:**
> "Test whether MNIST CNN features can transfer to text classification tasks using a character-level image representation."

**AI Output Assessment:**
- ✅ Creative experiment design.
- ❌ AI overclaimed that it "works" — the correct engineering answer is it does NOT transfer meaningfully because there is no shared visual structure between digit strokes and character encodings. Corrected the analysis.
