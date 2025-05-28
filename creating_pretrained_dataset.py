import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pathlib import Path
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models


def create_pretrained_dataset(input_dir, output_path):
    """
    Create a pretrained dataset from all CSV files in the input directory.
    Args:
        input_dir (str): Directory containing CSV files with SMILES data
        output_path (str): Path to save the combined SMILES dataset
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(Path(input_dir).glob('*.csv'))
    all_smiles = []
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            print(f"Processing {csv_file.name}...")
            df = pd.read_csv(csv_file)
            
            # Handle different column names for SMILES
            if csv_file.name == 'bace.csv':
                smiles_data = df['mol']
            elif 'smiles' in df.columns:
                smiles_data = df['smiles']
            else:
                print(f"Warning: No SMILES column found in {csv_file.name}")
                continue
                
            all_smiles.extend(smiles_data.tolist())
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
    
    # Remove duplicates while preserving order
    unique_smiles = list(dict.fromkeys(all_smiles))
    
    # Save all unique SMILES to a single file
    pd.Series(unique_smiles).to_csv(output_path, index=False, header=False)
    
    print(f"\nProcessing complete!")
    print(f"Total SMILES strings found: {len(all_smiles)}")
    print(f"Unique SMILES strings: {len(unique_smiles)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    create_pretrained_dataset('dataset/raw_dataset', 'dataset/pretrain_dataset/combined_smiles.txt')
    