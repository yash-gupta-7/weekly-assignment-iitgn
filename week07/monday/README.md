## git link : 
https://github.com/yash-gupta-7/weekly-assignment-iitgn/tree/main/week07/monday

# Week 07: NLP Foundations · TF-IDF from Scratch

## Project Title
**ShopSense Review Analysis using Ground-Up TF-IDF and Analytical Implementations**

## Problem Statement
The goal of this assignment is to understand the mechanics of NLP by implementing TF-IDF, Inverse Document Frequency scaling, and Cosine Similarities completely from scratch (without utilizing scikit-learn). Mathematical proofs must be shown to validate term domain uniqueness against baseline implementations. 

## Approach
1. **Modular Architecture:** Created `src` models containing purely testable, non-spaghetti code implementing custom document term matrices using Scipy Sparse Data Structures.
2. **Data Generation:** Utilized python stochastic selection mimicking actual E-commerce domains to generate `ShopSense Reviews`. 
3. **From Scratch Embeddings:** `tfidf_engine.py` generates full corpus IDFs, Euclidean constraints, and cosine rankers.
4. **Analytical Justification:** Implemented analytical TF-IDF logic in `analytical_models.py` for specific `clothing` indices, comparing ubiquitous tokens like "the" to rare occurrences like "embroidery".
5. **Bonus Modeling:** Built a BM25 pipeline for comparison against vanilla TF-IDF.

## Technologies Used
- Python 3.9
- Scipy (Sparse Matrices)
- Pandas / Numpy (Data Processing layer)
- Jupyter Notebook / nbformat (Pipeline orchestration layer)
- sklearn (`TfidfVectorizer` used exclusively for benchmarking average L2 differences)

## Results Summary
- **Vocabulary:** Generated vocabulary dimensions successfully against a completely un-imported mathematical framework. 
- **Error vs. Library implementations:** Scored incredibly low difference thresholds compared to default Sklearn structures `<0.01` L2 error.
- **Top Metrics:** Verified why niche terms rank exponentially higher than stop words due to specific domain distributions reducing corpus document frequencies.

## How to Run
1. Navigate into repository:
```bash
cd week07/monday
```
2. Set up virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn notebook nbclient nbformat scipy
```
3. Generate data:
```bash
python3 src/data_generator.py
```
4. Build the notebook:
```bash
python3 src/notebook_builder.py
```
5. Execute the completed `.ipynb` pipeline:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/week07_monday_assignment.ipynb
```

---
## 🔗 Git Resource
- **Project Repository**: [GitHub Link](https://github.com/yash-gupta-7/weekly-assignment-iitgn)

