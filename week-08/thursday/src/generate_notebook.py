import nbformat as nbf

nb = nbf.v4.new_notebook()

# Title
nb.cells.append(nbf.v4.new_markdown_cell("# Week 8 Thursday: Sequential Data Analysis & Modeling\n**Author:** Antigravity AI\n\nThis notebook addresses the sequential data challenges for Vikram Anand's fintech firm."))

# Imports
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import warnings
warnings.filterwarnings('ignore')"""))

# Q1: Stock Data Preparation
nb.cells.append(nbf.v4.new_markdown_cell("## Sub-step 1: Stock Data Preparation\nConstructing a sequence dataset for next-day close price prediction on RELIANCE."))

nb.cells.append(nbf.v4.new_code_cell("""# 1. Load and Prepare
df_stock = pd.read_csv('data/stock_prices.csv')
df_rel = df_stock[df_stock['ticker'] == 'RELIANCE'].copy()
df_rel['date'] = pd.to_datetime(df_rel['date'])
df_rel = df_rel.sort_values('date')

# 2. Sequential Windowing
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df_rel['close'].values.reshape(-1, 1))

WINDOW_SIZE = 30
X, y = create_sequences(scaled_close, WINDOW_SIZE)

# 3. Time-Series Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train shapes: {X_train.shape}, Test shapes: {X_test.shape}")"""))

nb.cells.append(nbf.v4.new_markdown_cell("""### Justification for Data Prep Decisions
- **Window Size (30):** A 30-day window captures approximately 6 weeks of trading data, allowing the model to learn medium-term trends and support/resistance levels.
- **Split Strategy (Time-Series Split):** For time-series data, we must use chronological splitting. Using a random split would result in **look-ahead bias**, where the model learns from future data to predict the past, leading to artificially inflated (and unrealistic) performance metrics.
- **Consequence of Random Split:** If a random split were used, the reported performance would be extremely high due to data leakage, but the model would fail completely when deployed in a real-time environment."""))

# Q2: Chat Data Preparation
nb.cells.append(nbf.v4.new_markdown_cell("## Sub-step 2: Chat Data Preparation\nResolving timestamp inconsistencies and conducting EDA on churn signals."))

nb.cells.append(nbf.v4.new_code_cell("""# 1. Load and Fix Timestamps
df_chat = pd.read_csv('data/chat_logs.csv')

def parse_inconsistent_date(date_str):
    try:
        return parser.parse(date_str)
    except:
        return np.nan

df_chat['timestamp'] = df_chat['timestamp'].apply(parse_inconsistent_date)
df_chat = df_chat.dropna(subset=['timestamp'])

# 2. EDA - Churn Signal Analysis
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='churn_within_30_days', y='sentiment_score', data=df_chat)
plt.title('Sentiment Score vs Churn')

plt.subplot(1, 2, 2)
sns.boxplot(x='churn_within_30_days', y='message_length', data=df_chat)
plt.title('Message Length vs Churn')
plt.show()

correlation = df_chat[['message_length', 'sentiment_score', 'churn_within_30_days']].corr()
print("Correlation with Churn:\\n", correlation['churn_within_30_days'])"""))

# Q3: LSTM for Stock
nb.cells.append(nbf.v4.new_markdown_cell("## Sub-step 3: LSTM Stock Price Prediction\nBuilding a sequence model for Indian equities."))

nb.cells.append(nbf.v4.new_code_cell("""class StockLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = StockLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Convert to Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Training Loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_t).numpy()

# Invert Scaling
y_test_orig = scaler.inverse_transform(y_test)
preds_orig = scaler.inverse_transform(predictions)

rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
mae = mean_absolute_error(y_test_orig, preds_orig)
print(f"\\nPerformance Metrics:\\nRMSE: {rmse:.2f}\\nMAE: {mae:.2f}")"""))

nb.cells.append(nbf.v4.new_markdown_cell("""### Architectural Decisions
- **LSTM (Long Short-Term Memory):** Chosen for its ability to mitigate the vanishing gradient problem in long sequences, making it suitable for capturing temporal dependencies in financial data.
- **Metric Selection:** RMSE is chosen as it penalizes larger errors more heavily, which is critical in trading where large deviations can lead to significant financial loss.
- **Deployment Threshold:** A model is worth deploying if its RMSE is significantly lower than a simple Naive model (persistence model) and if its directional accuracy (predicting price movement direction correctly) exceeds 55-60%, considering transaction costs."""))

# Q4 & Q5: Churn & Cost Model
nb.cells.append(nbf.v4.new_markdown_cell("## Sub-step 4 & 5: Churn Prediction & Cost Model\nComparing models and determining outreach strategy."))

nb.cells.append(nbf.v4.new_code_cell("""# Sub-step 4: Tabular vs Sequential
# Since the dataset is small and interactions are independent in our synthetic data, 
# we aggregate features per customer.
df_agg = df_chat.groupby('customer_id').agg({
    'sentiment_score': 'mean',
    'message_length': 'mean',
    'churn_within_30_days': 'max'
})

X_churn = df_agg.drop('churn_within_30_days', axis=1)
y_churn = df_agg['churn_within_30_days']

X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_c_train, y_c_train)
y_pred = clf.predict_proba(X_c_test)[:, 1]

# Sub-step 5: Cost Model
cost_false_pos = 50   # Cost of outreach to a happy customer
cost_false_neg = 500  # Cost of losing a customer (LTV)

thresholds = np.linspace(0, 1, 100)
costs = []

for t in thresholds:
    y_p = (y_pred >= t).astype(int)
    fp = ((y_p == 1) & (y_c_test == 0)).sum()
    fn = ((y_p == 0) & (y_c_test == 1)).sum()
    total_cost = (fp * cost_false_pos) + (fn * cost_false_neg)
    costs.append(total_cost)

optimal_threshold = thresholds[np.argmin(costs)]
print(f"Optimal Threshold for Outreach: {optimal_threshold:.2f}")

# Ranked Risk List
risk_list = pd.DataFrame({'customer_id': X_churn.index, 'churn_prob': clf.predict_proba(X_churn)[:,1]})
risk_list = risk_list.sort_values(by='churn_prob', ascending=False)
print("\\nTop 5 At-Risk Customers:\\n", risk_list.head())"""))

# Q6: LSTM vs Autoregressive
nb.cells.append(nbf.v4.new_markdown_cell("## Sub-step 6: LSTM vs. Autoregressive Baseline\nChallenging the complexity of neural networks."))

nb.cells.append(nbf.v4.new_code_cell("""# Simple Autoregressive Baseline (k-day moving average)
def moving_average_baseline(data, k=5):
    preds = []
    for i in range(len(data)):
        # Average of previous k values
        window = data[max(0, i-k):i]
        preds.append(np.mean(window) if len(window) > 0 else data[i])
    return np.array(preds)

# Evaluation on test set
# We use the previous window of the test set
test_data_raw = scaled_close[split_idx + WINDOW_SIZE - 5 : ] 
ar_preds = moving_average_baseline(test_data_raw, k=5)
ar_preds = ar_preds[5:] # align with test set

ar_preds_orig = scaler.inverse_transform(ar_preds.reshape(-1, 1))
ar_rmse = np.sqrt(mean_squared_error(y_test_orig, ar_preds_orig))

print(f"LSTM RMSE: {rmse:.2f}")
print(f"AR Baseline RMSE: {ar_rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_orig, label='Actual Price')
plt.plot(preds_orig, label='LSTM Predictions')
plt.plot(ar_preds_orig, label='AR Baseline', linestyle='--')
plt.legend()
plt.title('LSTM vs AR Baseline')
plt.show()"""))

# Q7: Manual BPTT
nb.cells.append(nbf.v4.new_markdown_cell("## Sub-step 7: Manual BPTT Implementation\nImplementing backpropagation through time from scratch."))

nb.cells.append(nbf.v4.new_code_cell("""class ManualRNN:
    def __init__(self, input_size, hidden_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        
    def forward(self, x_seq):
        h = np.zeros((self.W_hh.shape[0], 1))
        hs = { -1: h }
        xs = {}
        for t in range(len(x_seq)):
            xs[t] = x_seq[t].reshape(-1, 1)
            h = np.tanh(np.dot(self.W_xh, xs[t]) + np.dot(self.W_hh, h))
            hs[t] = h
        return hs, xs

    def backward(self, hs, xs, d_h_final):
        # Simplfied BPTT for weight gradients
        dW_xh, dW_hh = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh)
        dh_next = d_h_final
        
        for t in reversed(range(len(xs))):
            dtanh = (1 - hs[t] * hs[t]) * dh_next
            dW_xh += np.dot(dtanh, xs[t].T)
            dW_hh += np.dot(dtanh, hs[t-1].T)
            dh_next = np.dot(self.W_hh.T, dtanh)
            
        return dW_xh, dW_hh

# Gradient Magnitude vs Sequence Length Demo
rnn = ManualRNN(1, 10)
seq_lengths = [5, 10, 20, 50]
grad_norms = []

for L in seq_lengths:
    x_toy = np.random.randn(L, 1)
    hs, xs = rnn.forward(x_toy)
    dW_xh, _ = rnn.backward(hs, xs, np.ones((10, 1)))
    grad_norms.append(np.linalg.norm(dW_xh))

plt.plot(seq_lengths, grad_norms, marker='o')
plt.xlabel('Sequence Length')
plt.ylabel('Gradient Magnitude')
plt.title('Vanishing Gradient Demonstration')
plt.show()"""))

with open('notebooks/thursday_assignment.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
