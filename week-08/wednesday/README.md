# Week 08 Wednesday: CNNs & Semantic Search

## Project Overview
This project implements a multi-stage content moderation pipeline and a visual benchmark using CNNs. It covers data quality auditing, deep learning model construction, and semantic similarity search using sentence embeddings.

## Problem Statement
Meera Nair, Head of Trust & Safety, requires:
1. A **Content Classifier** to identify harmful posts (hate speech).
2. A **Semantic Search System** to detect coordinated campaigns by surfacing conceptually similar posts.

## Approach
- **Data Characterization**: Audited `social_media_posts.csv` and MNIST for distributions and quality issues.
- **CNN Development**: Built a 2-layer CNN on MNIST to understand learned visual representations.
- **Hate Speech Detection**: Implemented a classifier addressing class imbalance and data sparsity.
- **Semantic Retrieval**: Used Sentence-BERT (`MiniLM`) to surface semantically similar posts, comparing it against traditional TF-IDF.
- **Two-Stage Pipeline**: Combined classification and retrieval for robust moderation.

## Technologies Used
- **Deep Learning**: PyTorch (CNN)
- **Embeddings**: Sentence-Transformers (Sentence-BERT)
- **NLP**: Scikit-Learn (TF-IDF), Pandas
- **Visualization**: Matplotlib, Seaborn

## Results Summary
- The CNN achieved high accuracy on MNIST, with filters learning edge and stroke detection.
- Semantic search successfully surfaced related posts that shared no keywords, outperforming TF-IDF in conceptual matching.
- The two-stage pipeline significantly increased the detection of harmful content compared to a standalone classifier.

## How to Run
1. Create a virtual environment: `python3 -m venv .venv`
2. Activate it: `source .venv/bin/activate`
3. Install requirements: `pip install torch torchvision sentence-transformers pandas scikit-learn matplotlib seaborn nbformat`
4. Run the data preparation script: `python src/prepare_data.py`
5. Open and execute the notebook in `notebooks/Assignment_W8_Wednesday.ipynb`.
