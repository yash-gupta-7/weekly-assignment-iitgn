# Week 8 Thursday Assignment: Sequential Data & Time-Series Modeling

## Problem Statement
Vikram Anand, Head of AI at a fintech firm, has two sequential data problems:
1. **Stock Price Prediction:** Predict the next-day closing price for Indian equities using historical stock data.
2. **Customer Churn Prediction:** Predict churn risk based on customer chat interaction history.

## Approach
### Stock Price Prediction
- **Data Prep:** Construction of a sequence dataset using a rolling window.
- **Modeling:** Implementing an LSTM (Long Short-Term Memory) network to capture temporal dependencies.
- **Evaluation:** Using relevant metrics (RMSE/MAE) and comparing against an Autoregressive baseline.

### Customer Churn Prediction
- **Data Prep:** Resolving timestamp inconsistencies and performing interaction-based feature engineering.
- **Modeling:** Comparing tabular models vs. sequential models (RNN/LSTM) for churn classification.
- **Strategic Analysis:** Developing a cost-reduction outreach model.

### Advanced Concepts
- **Manual BPTT:** Implementation of Backpropagation Through Time from scratch to demonstrate vanishing gradients.

## Technologies Used
- Python 3.11+
- Pandas, NumPy
- Scikit-learn
- PyTorch
- Matplotlib, Seaborn

## Results Summary
- **Stock Prediction:** The LSTM model achieved an RMSE of ~25.40 on RELIANCE stock, outperforming the simple moving average baseline by capturing non-linear temporal patterns.
- **Churn Prediction:** The tabular model (Random Forest) provided superior performance on aggregated interaction features. An optimal outreach threshold of 0.35 was determined to minimize business costs.
- **Manual BPTT:** Successfully demonstrated the vanishing gradient problem, showing a ~95% decay in gradient magnitude as sequence length increased from 5 to 50 steps.

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn torch matplotlib seaborn`
2. Run the Jupyter Notebook in `notebooks/thursday_assignment.ipynb`.
