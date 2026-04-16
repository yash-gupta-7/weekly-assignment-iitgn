# Hospital Readmission prediction · Week 08 Tuesday

## Project Overview
This project implements a clinical decision support system for 30-day hospital readmission prediction, built for Dr. Priya Anand. The solution includes a rigorous data quality audit, principled cleaning, and a pure NumPy implementation of a 3-layer Neural Network.

## Problem Statement
Real-world hospital datasets are often messy (inconsistent labeling, missing values, outliers). This assignment focuses on:
1. Identifying and fixing data quality issues in patient records.
2. Building a deep learning model from first principles (NumPy) to predict readmission.
3. Optimizing the model's operating point based on clinical cost structures.

## Technology Stack
- **Language:** Python 3.9+
- **Mathematics:** NumPy (for NN from scratch)
- **Data Handling:** Pandas
- **Visualization:** Matplotlib, Seaborn
- **Baseline:** Scikit-Learn (Logistic Regression for comparison)

## Key Results
- **Neural Network:** Successfully implemented forward/backward propagation with Relu/Sigmoid activations.
- **Data Quality:** Audited `Age` and `BMI` columns, identifying multiple issues including missing values and inconsistent gender labeling.
- **Cost Optimization:** Identified an optimal clinical threshold (approx. 0.3) that balances the high cost of False Negatives ($500) against False Positives ($100).
- **The Accuracy Trap:** Demonstrated why 94% accuracy can be misleading in imbalanced healthcare datasets and used F1-score/Confusion Matrix as better metrics.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn notebook
   ```
2. Navigate to the `notebooks/` directory.
3. Open and run `hospital_readmission_analysis.ipynb`.

## Folder Structure
- `/data`: Contains `hospital_records.csv` and assignment documentation.
- `/src`: Modular Python scripts for data cleaning and model building.
- `/notebooks`: Fully executed analysis notebook.
- `prompts.md`: AI dialogue and critique as per course policy.
