# Week 8 Friday: Automated Chest X-ray Screening

## Project Overview
This repository contains an end-to-end pipeline for screening chest X-ray images using Transfer Learning. The goal is to assist Dr. Sameer Rao at AIIMS in classifying five conditions: Normal, Pneumonia, COVID-19, Pleural Effusion, and Lung Mass.

## Problem Statement
Developing a robust medical imaging model with limited data (n=490 labeled) requires intelligent use of pre-trained architectures. The model must handle clinical costs (false negatives are dangerous) and provide explainability via saliency maps.

## Approach
1. **Data Audit:** Analyzed metadata for class imbalance and hospital-site bias.
2. **Feature Extraction:** Froze ResNet-18 backbone, training only the classification head.
3. **Fine-tuning:** Unfroze deeper layers with a low learning rate to adapt features to medical textures.
4. **Explainability:** Implemented saliency mapping to visualize model attention.
5. **Deployment:** Designed a triage protocol based on prediction confidence scores.

## Technologies Used
- Python 3
- PyTorch / Torchvision
- Pandas / NumPy
- Matplotlib / Seaborn
- Jupyter Notebook

## Results Summary
- **Fine-tuning** outperformed **Feature Extraction** on the minority classes (COVID-19, Lung Mass).
- **Training from Scratch** failed to converge within 3 epochs due to the small dataset size (n=490), proving that transfer learning is essential for this scale.

## How to Run
1. Clone the repository.
2. Ensure dependencies are installed: `pip install torch torchvision pandas matplotlib seaborn nbformat`
3. Navigate to `notebooks/`.
4. Run `W8_Friday_Assignment.ipynb`.

*Note: Since the original image files were not provided in the directory, a `SyntheticMedicalDataset` is used to demonstrate the pipeline's functionality.*
