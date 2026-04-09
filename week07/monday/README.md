# Week 07 - Monday - AI/NLP Cohort 1

## Overview
This folder contains the Day 1 (Monday) solutions for the Week 07 Take-Home Assignment, focusing on TF-IDF, Count Vectorizers, and Text Similarity from scratch.

## Files
- `src/data_generator.py`: Generates the simulated `ShopSense_Reviews.csv` dataset.
- `src/create_notebook.py`: Automatically generates and structues the answers into a `.ipynb` format.
- `notebooks/week07_monday_assignment.ipynb`: The finally executed notebook containing equations, visualizations, and markdown answers to Q1 and Q2.

## How to Install Dependencies
```bash
pip install pandas numpy scikit-learn notebook nbclient nbformat scipy
```

## How to Run Local Data Generation & Code
If you want to reproduce everything from scratch:
1. Ensure you are in the repository root and `mkdir -p week07/monday/data`
2. Run `python week07/monday/src/data_generator.py` to synthesize the 10K test data.
3. Run `python week07/monday/src/create_notebook.py` to compile the notebook.
4. Run `jupyter nbconvert --to notebook --execute --inplace week07/monday/notebooks/week07_monday_assignment.ipynb` to execute it completely.

## Expected Output
Running the notebook will output the vocabulary size, average TF-IDF errors vs `sklearn` vectorizer, Top-5 matched queries through `cosine_similarity,` as well as answers to Q2 using BM25 and manual IDF comparisons.
