# AI Usage Prompts - Week 08 Tuesday

## Task: Data Cleaning and Neural Network from Scratch

### Prompt 1: Modular Data Cleaner
"Create a python module `data_cleaner.py` that contains an `audit_data` function and a `clean_data` function. `audit_data` should check for missing values, outliers in age and BMI, and inconsistencies in gender and department columns. `clean_data` should normalize categories, impute missing values with medians, and perform one-hot encoding."

**Critique:** The output was correct and structured. I modified the `clean_data` function to include date feature extraction which wasn't explicitly in the prompt but is best practice.

### Prompt 2: NumPy NN Implementation
"Wait, build a 3-layer neural network (2 hidden layers) in pure NumPy for binary classification. Include ReLU for hidden layers and Sigmoid for output. Must include forward, BCE loss, and backpropagation. Use variable names consistent with standard ML literature (W1, b1, z1, a1, etc.)."

**Critique:** The initial output used a simple gradient descent. I added Xavier initialization to ensure better convergence, as deep networks can suffer from vanishing/exploding gradients.

### Prompt 3: Jupyter Notebook Structure
"Create a Jupyter Notebook that follows the sub-steps in the W8 Tuesday assignment: Audit, Clean, Build NN, Train & Evaluate, and Cost Optimization. Use `nbformat` to generate the notebook programmatically."

**Critique:** Very efficient for generating a structured assignment. I manually adjusted the cost assumptions (False Negative = $500, False Positive = $100) to reflect realistic clinical settings where missing a patient is significantly more expensive.
