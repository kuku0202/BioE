import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random

def split_data(input_file, output_dir='dataset/splits', train_ratio=0.8, val_ratio=0.05, test_ratio=0.15, random_state=42):
    """
    Split the data into train, validation, and test sets using pandas.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to save the split files (default: 'dataset/splits')
        train_ratio (float): Proportion of data for training (default: 0.8)
        val_ratio (float): Proportion of data for validation (default: 0.05)
        test_ratio (float): Proportion of data for testing (default: 0.15)
        random_state (int): Random seed for reproducibility (default: 42)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # First split: separate out the test set
    train_val, test = train_test_split(
        df, 
        test_size=test_ratio,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    # Adjust the validation ratio to account for the remaining data
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state
    )
    
    # Save the splits to separate CSV files
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data split complete:")
    print(f"Training set: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    print(f"Files saved in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    input_file = "dataset/delaney-processed.csv"
    split_data(input_file)