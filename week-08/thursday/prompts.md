# AI Prompts & Critique

## Sub-step 1 & 2: Data Preprocessing
**Prompt:** "construct a sequence dataset for next-day close price prediction on one stock of your choice. document window size, split strategy and justify. handle inconsistent timestamps in chat_logs.csv and conduct EDA on churn signal."

**Critique:** The AI correctly suggested a chronological split and a rolling window. I modified the window size to 30 to capture seasonal trends and used `dateutil.parser` for the messy timestamps.

## Sub-step 3: LSTM Stock Prediction
**Prompt:** "Build an LSTM to predict next-day stock closing price using the sequence dataset. justify architectural decisions and report performance."

**Critique:** The AI proposed a multi-layer LSTM. I added dropout to prevent overfitting and chose RMSE as the primary metric for its sensitivity to large errors in trading scenarios.

## Sub-step 4 & 5: Churn Model & Cost Strategy
**Prompt:** "Build a model to predict customer churn. compare tabular vs sequential. produce a ranked risk list and define a cost model for outreach."

**Critique:** The AI suggested a Random Forest for the tabular approach. I implemented a threshold analysis based on financial costs (False Positives vs False Negatives), which is more meaningful than pure accuracy for business stakeholders.

## Sub-step 6 & 7: Hard Tasks
**Prompt:** "Implement manual BPTT and compare LSTM against an AR baseline. demonstrate vanishing gradients."

**Critique:** The manual BPTT derivation was mostly correct, but I had to refine the gradient flow through the tanh activation. The vanishing gradient demonstration clearly shows the magnitude decay, which validates the need for LSTMs.
