import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_stock_data(filepath, ticker='RELIANCE', window_size=30, test_size=0.2):
    df = pd.read_csv(filepath)
    df = df[df['ticker'] == ticker].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    data = df['close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size])
        
    X, y = np.array(X), np.array(y)
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

class StockLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, X_train, y_train, epochs=20, lr=0.001):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_stock_data('data/stock_prices.csv')
    model = StockLSTM()
    model = train_model(model, X_train, y_train)
    
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test_t).numpy()
    
    preds_orig = scaler.inverse_transform(preds)
    y_test_orig = scaler.inverse_transform(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mae = mean_absolute_error(y_test_orig, preds_orig)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
