# Week 07 Friday: Production Sentiment Deployment · Cost & Constraints

## Project Title
**ShopSense Multi-Constraint Sentiment Classifier: Scaling for 100K Reviews/Day**

## Problem Statement
Moving from a laboratory model to production requires balancing technical performance (F1-score) against hard engineering constraints: inference speed (<20ms), language diversity (15% Hinglish), and stability on new categories. Critically, we must justify the deployment using a **Business Cost Model** rather than accuracy alone, accounting for the financial impact of missed churners.

## Approach
1. **Imbalanced Training:** Implemented a stratifying pipeline in `sentiment_classifier.py` to handle the 70/10 skew of 5-star vs 1-star reviews.
2. **Constraint Quantification:** `constraint_evaluator.py` measures real-world metrics—inference latency and Hinglish accuracy—to ensure the model meets ShopSense's SLA.
3. **Financial Impact Modeling:** Developed a cost model in `cost_analyzer.py` predicting that False Negatives (missed 1-star reviews) cost the business 25x more than False Positives ($50 vs $2).
4. **Monitoring & Governance:** Defined a technical brief for Priya including retraining thresholds (F1 < 0.85) to prevent "model drift" before it triggers customer complaints.

## Technologies Used
- **Python 3.9**
- **Scikit-Learn** (Logistic Regression, TfidfVectorizer)
- **Matplotlib / Seaborn** (Data imbalance visualization)
- **Jupyter Notebook** (`nbconvert` for production-grade execution)

## Results Summary
- **The Accuracy Trap:** Demonstrated how a broken model reporting 94% accuracy can actually have **0% Recall** for 1-star complaints, costing massive revenue loss.
- **Production Choice:** Recommended Logistic Regression due to its <1ms inference speed and robust performance on Hinglish without needing heavy DL infrastructure.
- **Revenue Protection:** Applied cost analysis projections showing exactly why Precision/Recall balance is more profitable than raw accuracy.

## How to Run
1. Navigate to directory:
```bash
cd week07/friday/
```
2. Setup environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn notebook nbconvert nbformat scipy matplotlib seaborn
```
3. Generate dataset:
```bash
python3 src/data_generator.py
```
4. Build and execute the pipeline:
```bash
python3 src/notebook_builder.py
jupyter nbconvert --to notebook --execute --inplace notebooks/week07_friday_assignment.ipynb
```
