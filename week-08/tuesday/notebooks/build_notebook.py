import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Title
nb['cells'].append(nbf.v4.new_markdown_cell("# Week 08 · Tuesday Assignment: Deep Learning + Data Cleaning\n"
                                             "**Goal:** Audit and clean hospital data, and build a neural network from scratch to predict 30-day readmission.\n"
                                             "**Scenario:** Dr. Priya Anand's readmission prediction system."))

# Sub-step 1
nb['cells'].append(nbf.v4.new_markdown_cell("## Sub-step 1: Data Quality Audit\n"
                                             "In this step, we examine the `hospital_records.csv` for data quality issues, specifically focusing on Age and BMI columns."))

nb['cells'].append(nbf.v4.new_code_cell("import pandas as pd\n"
                                        "import numpy as np\n"
                                        "import matplotlib.pyplot as plt\n"
                                        "import seaborn as sns\n"
                                        "import sys\n"
                                        "import os\n"
                                        "sys.path.append('../src')\n"
                                        "from data_cleaner import audit_data, clean_data\n\n"
                                        "# Load data\n"
                                        "df = pd.read_csv('../data/hospital_records.csv')\n"
                                        "print(f'Dataset shape: {df.shape}')\n"
                                        "df.head()"))

nb['cells'].append(nbf.v4.new_code_cell("# Run audit\n"
                                        "issues = audit_data(df)\n"
                                        "print('Data Quality Issues Found:')\n"
                                        "import json\n"
                                        "print(json.dumps(issues, indent=4))"))

# Sub-step 2
nb['cells'].append(nbf.v4.new_markdown_cell("## Sub-step 2: Principled Data Cleaning\n"
                                             "Applying fixes for the documented issues: normalizing gender, handling missing BMI, and encoding categorical variables."))

nb['cells'].append(nbf.v4.new_code_cell("df_clean = clean_data(df)\n"
                                        "print(f'Cleaned dataset shape: {df_clean.shape}')\n"
                                        "df_clean.head()"))

# Sub-step 3
nb['cells'].append(nbf.v4.new_markdown_cell("## Sub-step 3: 3-Layer Neural Network in NumPy\n"
                                             "Implementing forward prop, loss function (BCE), and backprop from scratch."))

nb['cells'].append(nbf.v4.new_code_cell("from nn_numpy import SimpleNN\n"
                                        "from sklearn.model_selection import train_test_split\n"
                                        "from sklearn.preprocessing import StandardScaler\n\n"
                                        "# Prepare data\n"
                                        "X = df_clean.drop(columns=['readmitted_30d']).values\n"
                                        "y = df_clean['readmitted_30d'].values.reshape(-1, 1)\n\n"
                                        "# Split data\n"
                                        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\n"
                                        "# Scale data\n"
                                        "scaler = StandardScaler()\n"
                                        "X_train_scaled = scaler.fit_transform(X_train)\n"
                                        "X_test_scaled = scaler.transform(X_test)\n\n"
                                        "# Initialize Model\n"
                                        "input_size = X_train_scaled.shape[1]\n"
                                        "nn = SimpleNN(input_size=input_size, hidden_sizes=[32, 16], learning_rate=0.05)"))

# Sub-step 4
nb['cells'].append(nbf.v4.new_markdown_cell("## Sub-step 4: Training and Evaluation\n"
                                             "Training the network and comparing with a standard sklearn classifier (Logistic Regression)."))

nb['cells'].append(nbf.v4.new_code_cell("history = nn.train(X_train_scaled, y_train, epochs=2000)\n\n"
                                        "# Plot loss\n"
                                        "plt.figure(figsize=(10, 5))\n"
                                        "plt.plot(history)\n"
                                        "plt.title('Training Loss Card')\n"
                                        "plt.xlabel('Epochs')\n"
                                        "plt.ylabel('Loss')\n"
                                        "plt.show()"))

nb['cells'].append(nbf.v4.new_code_cell("from sklearn.metrics import classification_report, confusion_matrix, f1_score\n\n"
                                        "# Predictions\n"
                                        "y_pred_proba = nn.forward(X_test_scaled)\n"
                                        "y_pred = (y_pred_proba > 0.5).astype(int)\n\n"
                                        "print('NumPy NN Classification Report:')\n"
                                        "print(classification_report(y_test, y_pred))\n\n"
                                        "# sklearn comparison\n"
                                        "from sklearn.linear_model import LogisticRegression\n"
                                        "lr_model = LogisticRegression()\n"
                                        "lr_model.fit(X_train_scaled, y_train.ravel())\n"
                                        "y_pred_lr = lr_model.predict(X_test_scaled)\n"
                                        "print('\\nSklearn Logistic Regression Classification Report:')\n"
                                        "print(classification_report(y_test, y_pred_lr))"))

# Sub-step 5
nb['cells'].append(nbf.v4.new_markdown_cell("## Sub-step 5: Clinical Cost Optimization\n"
                                             "Finding the optimal threshold to minimize expected clinical cost.\n"
                                             "**Assumptions:** False Negative (missing high-risk) = $500, False Positive (false alarm) = $100."))

nb['cells'].append(nbf.v4.new_code_cell("thresholds = np.linspace(0.01, 0.99, 100)\n"
                                        "costs = []\n"
                                        "fn_cost = 500\n"
                                        "fp_cost = 100\n\n"
                                        "for t in thresholds:\n"
                                        "    y_p = (y_pred_proba > t).astype(int)\n"
                                        "    cm = confusion_matrix(y_test, y_p)\n"
                                        "    if cm.shape == (2, 2):\n"
                                        "        tn, fp, fn, tp = cm.ravel()\n"
                                        "        total_cost = (fn * fn_cost) + (fp * fp_cost)\n"
                                        "        costs.append(total_cost)\n"
                                        "    else:\n"
                                        "        costs.append(np.inf)\n\n"
                                        "best_threshold = thresholds[np.argmin(costs)]\n"
                                        "print(f'Optimal clinical threshold: {best_threshold:.2f}')\n"
                                        "print(f'Minimum normalized cost: {min(costs)}')\n\n"
                                        "plt.figure(figsize=(10, 5))\n"
                                        "plt.plot(thresholds, costs)\n"
                                        "plt.axvline(best_threshold, color='red', linestyle='--')\n"
                                        "plt.title('Clinical Cost vs Threshold')\n"
                                        "plt.xlabel('Threshold')\n"
                                        "plt.ylabel('Total Cost ($)')\n"
                                        "plt.show()"))

# Sub-step 6
nb['cells'].append(nbf.v4.new_markdown_cell("## Sub-step 6: The 94% Accuracy Trap (Hard)\n"
                                             "Reproducing a pipeline that yields high accuracy but low utility due to class imbalance."))

nb['cells'].append(nbf.v4.new_code_cell("# If we just predict 0 (majority class)\n"
                                        "y_dummy = np.zeros_like(y_test)\n"
                                        "accuracy = (y_dummy == y_test).mean()\n"
                                        "print(f'Majority Class Classifier Accuracy: {accuracy*100:.2f}%')\n"
                                        "print('\\nConfusion Matrix (Majority Class):')\n"
                                        "print(confusion_matrix(y_test, y_dummy))\n"
                                        "print('\\nNote: High accuracy is misleading when class imbalance is present.')"))

# Sub-step 7
nb['cells'].append(nbf.v4.new_markdown_cell("## Sub-step 7: Neural Network as Feature Extractor (Hard)\n"
                                             "Extracting activations from the penultimate layer to use as patient embeddings, then training a Logistic Regression classifier on top."))

nb['cells'].append(nbf.v4.new_code_cell("# Extract activations\n"
                                        "def get_embeddings(nn, X):\n"
                                        "    nn.forward(X)\n"
                                        "    return nn.a2 # Penultimate layer activations\n\n"
                                        "X_train_embeddings = get_embeddings(nn, X_train_scaled)\n"
                                        "X_test_embeddings = get_embeddings(nn, X_test_scaled)\n\n"
                                        "print(f'Embedding shape: {X_train_embeddings.shape}')\n\n"
                                        "# Train classifier on embeddings\n"
                                        "from sklearn.linear_model import LogisticRegression\n"
                                        "clf_emb = LogisticRegression()\n"
                                        "clf_emb.fit(X_train_embeddings, y_train.ravel())\n\n"
                                        "# Evaluate\n"
                                        "y_pred_emb = clf_emb.predict(X_test_embeddings)\n"
                                        "print('Embedding-based Classifier Report:')\n"
                                        "print(classification_report(y_test, y_pred_emb))\n\n"
                                        "print('Explanation: The embeddings capture non-linear relationships learned by the early layers of the NN, allowing a linear classifier to separate classes more effectively in some cases.')"))

with open('hospital_readmission_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
