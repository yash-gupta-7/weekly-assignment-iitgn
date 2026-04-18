import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_chat_logs(n_rows=1000):
    np.random.seed(42)
    customer_ids = [f'C{i:04d}' for i in range(100)]
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    # Various timestamp formats as per the "inconsistent format" hint
    formats = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%m-%d-%y %I:%M %p", "%Y/%m/%d %H:%M:%S"]
    
    for _ in range(n_rows):
        cid = np.random.choice(customer_ids)
        days_offset = np.random.randint(0, 90)
        ts = start_date + timedelta(days=days_offset, minutes=np.random.randint(0, 1440))
        
        fmt = np.random.choice(formats)
        ts_str = ts.strftime(fmt)
        
        # Interaction patterns
        msg_len = np.random.randint(10, 500)
        sentiment = np.random.uniform(-1, 1)
        
        # Logic for churn (more likely if sentiment is low or msg_len is high)
        churn_prob = 0.1
        if sentiment < -0.5: churn_prob += 0.3
        if msg_len > 300: churn_prob += 0.2
        
        churned = 1 if np.random.random() < churn_prob else 0
        
        data.append({
            'timestamp': ts_str,
            'customer_id': cid,
            'message_length': msg_len,
            'sentiment_score': sentiment,
            'churn_within_30_days': churned
        })
        
    df = pd.DataFrame(data)
    df.to_csv('data/chat_logs.csv', index=False)
    print("Synthetic chat_logs.csv created.")

if __name__ == "__main__":
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    generate_synthetic_chat_logs()
