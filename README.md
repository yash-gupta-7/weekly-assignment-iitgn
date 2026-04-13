# Week 07: NLP Foundations · Embeddings · Sentiment · Production Strategy
### IIT Gandhinagar · Cohort 1 · Weekly Take-Home Assignment

This repository contains the end-to-end solutions for the **Week 07 NLP** daily assignments. The project is structured modularly, focusing on high-standard software engineering, mathematical rigor, and business-focused machine learning evaluation.

---

## 📂 Project Structure

The assignments are organized day-wise, each containing its own isolated dataset, source code, and executed Jupyter Notebook.

| Directory | Topic | Key Deliverables |
| :--- | :--- | :--- |
| [**Monday**](./week07/monday) | **TF-IDF from Scratch** | Custom sparse matrix engine, BM25 logic, and mathematical proofs. |
| [**Tuesday**](./week07/tuesday) | **Word2Vec & Semantic Gaps** | Polysemy disambiguation, Word2Vec vs. BERT comparison. |
| [**Wednesday**](./week07/wednesday) | **Hard NLP Patterns & ABSA** | Sarcasm/Code-mixing detection and Aspect-Based Sentiment Analysis. |
| [**Friday**](./week07/friday) | **Production & Cost Analysis** | Class imbalance strategy, Engineering constraints, and Financial cost modeling. |

---

## 🛠️ Technology Stack
- **Languages:** Python 3.9+
- **Core NLP:** Gensim (Word2Vec), Sentence-Transformers (BERT), Scikit-Learn.
- **Mathematics & Data:** SciPy (Sparse Matrices), NumPy, Pandas.
- **Visualization:** Matplotlib, Seaborn.
- **Orchestration:** Jupyter, `nbformat`, `nbconvert`.

---

## 🚀 Key Highlights

### 1. From-Scratch Engineering
Implemented a custom TF-IDF engine using SciPy's sparse data structures, achieving `<0.01` L2 error compared to standard libraries while proving the logarithmic behavior of domain-specific terms.

### 2. Semantic Analysis
Demonstrated the "Semantic Gap" where lexical models (BOW) score 0% similarity on synonymous reviews, while Deep Learning models (S-BERT) correctly identify high congruence.

### 3. Production Deployment Strategy
Developed a financial cost model for 100K reviews/day, justifying model selection based on the **Daily Misclassification Cost** (balancing False Negatives @ $50 vs False Positives @ $2).

---

## 📥 Local Setup & Execution
Each subfolder contains its own setup instructions. However, to run the entire week's pipeline from the root:

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn notebook nbconvert nbformat scipy gensim sentence-transformers matplotlib seaborn
   ```

2. **Navigate to a specific day:**
   ```bash
   cd week07/friday/
   ```

3. **Execute Pipeline:**  
   Most days follow the pattern: `data_generator.py` → `notebook_builder.py` → `nbconvert`.

---

## 🔗 Git Resources
- **Repository Root:** [GitHub Link](https://github.com/yash-gupta-7/weekly-assignment-iitgn)
- **Branch:** `main`
