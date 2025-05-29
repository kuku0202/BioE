# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import re

# class SMILESTokenizer:
#     def __init__(self):
#         # Basic SMILES tokens
#         self.tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
#         self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
#         self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
#         # Add common SMILES characters
#         smiles_chars = set('CNOSPFIHBrc=#()[]+-.\\/@123456789')
#         for char in smiles_chars:
#             if char not in self.token_to_idx:
#                 self.token_to_idx[char] = len(self.token_to_idx)
#                 self.idx_to_token[len(self.idx_to_token)] = char
    
#     def tokenize(self, smiles):
#         # Add special tokens
#         tokens = ['[CLS]'] + list(smiles) + ['[SEP]']
#         return tokens
    
#     def convert_tokens_to_ids(self, tokens):
#         return [self.token_to_idx.get(token, self.token_to_idx['[UNK]']) for token in tokens]
    
#     def convert_ids_to_tokens(self, ids):
#         return [self.idx_to_token[idx] for idx in ids]

# class SMILESDataset(Dataset):
#     def __init__(self, smiles_list, tokenizer, max_length=128):
#         self.smiles_list = smiles_list
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.smiles_list)
    
#     def __getitem__(self, idx):
#         smiles = self.smiles_list[idx]
#         tokens = self.tokenizer.tokenize(smiles)
        
#         # Truncate or pad to max_length
#         if len(tokens) > self.max_length:
#             tokens = tokens[:self.max_length-1] + ['[SEP]']
#         else:
#             tokens = tokens + ['[PAD]'] * (self.max_length - len(tokens))
        
#         # Convert tokens to ids
#         input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
#         # Create attention mask
#         attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        
#         return {
#             'input_ids': torch.tensor(input_ids),
#             'attention_mask': torch.tensor(attention_mask)
#         }

# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoder = nn.Dropout(dropout)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         self.decoder = nn.Linear(d_model, vocab_size)
    
#     def forward(self, src, src_mask=None):
#         src = self.embedding(src)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_mask)
#         output = self.decoder(output)
#         return output

# def train_model(model, train_loader, device, num_epochs=10):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
#         for batch in progress_bar:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
            
#             # Create target by shifting input_ids
#             target = input_ids[:, 1:].contiguous()
#             input_ids = input_ids[:, :-1]
            
#             optimizer.zero_grad()
#             output = model(input_ids, attention_mask[:, :-1])
            
#             loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
#         print(f'Epoch {epoch+1} average loss: {total_loss / len(train_loader)}')

# def main():
#     # Load SMILES data
#     smiles_data = pd.read_csv('dataset/pretrain_dataset/combined_smiles.txt', header=None)[0].tolist()
    
#     # Initialize tokenizer
#     tokenizer = SMILESTokenizer()
    
#     # Create dataset and dataloader
#     dataset = SMILESDataset(smiles_data, tokenizer)
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
#     # Initialize model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TransformerModel(vocab_size=len(tokenizer.token_to_idx)).to(device)
    
#     # Train model
#     train_model(model, train_loader, device)
    
#     # Save model
#     torch.save(model.state_dict(), 'dataset/pretrain_dataset/pretrained_model.pt')
#     print("Model saved to dataset/pretrain_dataset/pretrained_model.pt")

# if __name__ == "__main__":
#     main() 



