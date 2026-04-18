import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def load_and_characterize_data(file_path):
    """
    Loads medical imaging metadata and prints characterization stats.
    """
    df = pd.read_csv(file_path)
    
    # Label distribution
    label_counts = df['label'].value_counts(dropna=False)
    
    # Subgroup differences (Hospital Site)
    hospital_dist = pd.crosstab(df['hospital_site'], df['label'])
    
    # Image quality impact
    quality_dist = pd.crosstab(df['image_quality'], df['label'])
    
    return df, label_counts, hospital_dist, quality_dist

class SyntheticMedicalDataset(Dataset):
    """
    A synthetic dataset that generates random images based on metadata.
    Used when actual image files are not provided to ensure code execution.
    """
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        # Map labels to integers
        self.label_map = {l: i for i, l in enumerate(self.df['label'].dropna().unique())}
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        width = int(row['img_width_px'])
        # Generate a synthetic "X-ray" like image (random noise with some structure)
        # In a real scenario, this would load From row['image_id']
        image_np = np.random.randint(0, 256, (width, width), dtype=np.uint8)
        image = Image.fromarray(image_np).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_str = row['label']
        if pd.isna(label_str):
            label = -1 # Unlabeled
        else:
            label = self.label_map[label_str]
            
        return image, label, row['image_id']
