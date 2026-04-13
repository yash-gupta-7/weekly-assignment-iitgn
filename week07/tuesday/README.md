## git link : 
https://github.com/yash-gupta-7/weekly-assignment-iitgn/tree/main/week07/tuesday

# Week 07 Tuesday: NLP Embeddings, Sentiment & Model Drift

## Project Title
**ShopSense Semantic Analysis: Word2Vec, Polysemy, & Sentence Embeddings**

## Problem Statement
The goal is to analyze embeddings representations of words, testing their robustness against polysemy (e.g. the word "cheap" meaning both affordable and flimsy). Furthermore, to evaluate "semantic gaps" by scoring complex sentences against completely disjoint lexical counterparts ("incredible camera" vs "stunning photos") using progressively complex encoders from Bag of Words (BOW) scaling up to Sentence-BERT embeddings.

## Approach
1. **Data Generation:** Built synthetic reviews into `ShopSense_Reviews_Tuesday.csv` focusing heavily on polysemy contexts for the word "cheap".
2. **Word2Vec Modeling:** Designed `src/word2vec_models.py` to train local vector embeddings, analyze singular continuous vectors forced to represent ambiguous contexts, and disambiguate instances mapping contextual anchor embeddings.
3. **Similarity Gap Analysis:** Produced `src/similarity_models.py` constructing BOW matching, tiny TF-IDF evaluation, Word2Vec averaging mappings, and MiniLM Sentence-BERT encoding for robust comparisons.
4. **Jupyter Execution:** `notebook_builder.py` unifies operations into a mathematical breakdown output to the `.ipynb` file.

## Technologies Used
- **Python 3.9**
- **Gensim** (Word2Vec)
- **Sentence-Transformers** (all-MiniLM-L6-v2) 
- **Scipy / numpy / pandas**
- Jupyter Notebook (`nbconvert`)

## Results Summary
- **Word2Vec on Polysemy:** Identified mathematically that the embedding of an ambiguous word represents an intermediate centroid between its varied usages (cosine differences captured). Window sizes of 2 favor tight syntactical binds while `window=10` models capture broader topic similarities.
- **The Semantic Gap:** Demonstrated exactly why lexical-matching (BOW/TF-IDF) scores 0.0 on sentences describing identical domains using synonymous words, whereas Sentence-BERT successfully crosses the semantic divide providing high congruence embeddings scoring accurately against complex sentiment mixes ("incredible ... but terrible ..").

## How to Run Local Environment
1. Setup a fresh venv:
```bash
cd week07/tuesday/
python3 -m venv .venv
source .venv/bin/activate
pip install gensim sentence-transformers pandas numpy scikit-learn notebook scipy
```
2. Generate baseline metrics:
```bash
python3 src/data_generator.py
```
3. Run the orchestration pipeline:
```bash
python3 src/notebook_builder.py
jupyter nbconvert --to notebook --execute --inplace notebooks/week07_tuesday_assignment.ipynb
```
