import pandas as pd
import os
from pathlib import Path

# Create pretrain_dataset directory if it doesn't exist
pretrain_dir = Path('dataset/pretrain_dataset')
pretrain_dir.mkdir(parents=True, exist_ok=True)

# Get all CSV files from raw_dataset
raw_dataset_dir = Path('dataset/raw_dataset')
csv_files = list(raw_dataset_dir.glob('*.csv'))

# List to store all SMILES strings
all_smiles = []

# Process each CSV file
for csv_file in csv_files:
    try:
        print(f"Processing {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Check if 'smiles' column exists
        if 'smiles' in df.columns:
            smiles_data = df['smiles']
            all_smiles.extend(smiles_data.tolist())
        else:
            print(f"Warning: No 'smiles' column found in {csv_file.name}")
            
    except Exception as e:
        print(f"Error processing {csv_file.name}: {str(e)}")

# Remove duplicates while preserving order
unique_smiles = list(dict.fromkeys(all_smiles))

# Save all unique SMILES to a single file
output_file = pretrain_dir / 'combined_smiles.txt'
pd.Series(unique_smiles).to_csv(output_file, index=False, header=False)

print(f"\nProcessing complete!")
print(f"Total SMILES strings found: {len(all_smiles)}")
print(f"Unique SMILES strings: {len(unique_smiles)}")
print(f"Output saved to: {output_file}") 