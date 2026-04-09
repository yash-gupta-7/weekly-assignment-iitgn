# Weekly Assignment - IITGN Cohort 1 (NLP Foundations)

## Project Title
**Week 07: TF-IDF, Sentiment, & Embeddings (ShopSense Data)**

## Problem Statement
The goal is to implement fundamental Nature Language Processing (NLP) models *from scratch* — specifically TF-IDF — to rank relevant queries mathematically. The assignment strictly enforces vector operations without standard library pipelines like Scikit-Learn to ensure a deeper understanding of Document Frequencies (DF), Term Frequencies (TF), and Inverse Document Frequencies (IDF). It also asks to demonstrate and calculate by hand properties of specific domains (e.g. Stop words vs Niche domain terminology) and finally testing BM25 scaling parameters against baseline Cosine Similarities.

## Approach
1. **Mock Data Creation**: Developed a synthetic 10,000-row `ShopSense` Review dataset using probability-distributed category vocabulary injected across 6 semantic domains (Electronics, Clothing, Food, etc.)
2. **From-Scratch Matrices**: Used SciPy's sparse coordinate and compressed sparse row implementations to generate custom L2-normalized Term Frequency & IDF matrices.
3. **Similarity Search**: For the query `wireless earbuds battery life poor`, L2-normalized vector dot products were utilized to calculate cosine similarities.
4. **Validation**: Computed matrix difference utilizing `scipy.sparse.linalg.norm` and validated analytical word behaviors. Applied the BM25 variant as a bonus comparison over TF-IDF. 

## Technologies Used
- **Python 3**: Core execution
- **Pandas / NumPy**: Dataset tabularization, Vector operations
- **SciPy**: Sparse matrix logic (LIL, CSR) and Linear Algebra calculations
- **Jupyter Notebook**: Visualization and Executed Code reports
- **nbformat & nbclient**: Notebook auto-compilation structure.

## Results Summary
- Our manual TF-IDF implementation returned a vocabulary dimension consistent with zero average L2 error vs perfectly synchronized Sklearn tokenizers (`average L2 diff ~ 0.00`).
- Mathematical proofs confirm that ubiquitous words (`the`) have log evaluations driving IDF multipliers toward neutral ~`1.0` unboosted, whereas niche terms (`embroidery`) boost IDF magnitudes vastly due to log(N/DF).
- The BM25 algorithm reshuffles specific rankings due to integrated document-length penalizing and score saturation via the `k1` and `b` constants.

## How to Run Local Environment
1. Setup a fresh venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn notebook nbclient nbformat scipy
```
2. Navigate into `week07/monday`:
```bash
cd week07/monday/
```
3. Run the complete pipeline (including notebook generation):
```bash
mkdir -p data
python src/data_generator.py
python src/create_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/week07_monday_assignment.ipynb
```

Check `/week07/monday/notebooks/week07_monday_assignment.ipynb` to view the comprehensive execution and mathematically proven outputs directly from Jupyter.
