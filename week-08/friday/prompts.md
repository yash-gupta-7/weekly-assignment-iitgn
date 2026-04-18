# AI Usage Prompts & Critique

## Sub-step 1: Data Characterization
**Prompt:** "Write a Python function to load a CSV of medical metadata and produce a per-class distribution plot using Seaborn."
**Critique:** The AI provided a standard `value_counts()` and `countplot`. I modified it to include a cross-tabulation with `hospital_site` to address the "subgroup differences" requirement in the rubric.

## Sub-step 2 & 3: Transfer Learning
**Prompt:** "How to implement feature extraction vs fine-tuning in torchvision ResNet-18?"
**Critique:** The AI correctly suggested freezing `parameters()`. I encapsulated this into a modular `get_model` function in `src/model_training.py` with a `mode` switch for better engineering quality.

## Sub-step 4: Saliency Maps
**Prompt:** "Simple ResNet saliency map implementation in PyTorch."
**Critique:** Provided a basic gradient-based approach. It worked well. I added comments explaining what the model is attending to (e.g., edges and textures in the synthetic data).

## Sub-step 5: Synthesis
**Prompt:** "Explain LSTMs to a classmate focusing on gates and vanishing gradients. 200 words. 2 interview questions."
**Critique:** The explanation was clear. I replaced some generic terms with more technical descriptions of the tanh and sigmoid activations to show deeper understanding.

## Sub-step 6 & 7: Hard Tasks
**Prompt:** "Design a confidence-based triage protocol for clinical AI deployment."
**Critique:** The AI suggested basic thresholds. I formalized these into three tiers (Auto, Review, Reject) to match the sub-step 7 requirements.
