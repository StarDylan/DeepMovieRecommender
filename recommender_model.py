import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import csv

# 1. Load the datasets
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv', low_memory=False)

# Create a dictionary mapping movie IDs to their titles
movie_id_to_title = {}
with open('data/movies.csv', 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for row in reader:
        movie_id = int(row[0])
        title = row[1]
        movie_id_to_title[movie_id] = title

# Merge the datasets
data = pd.merge(movies, ratings, on='movieId')

# Convert user and item IDs to integers (index-based)
user_ids = {id: i for i, id in enumerate(data['userId'].unique())}
movie_ids = {id: i for i, id in enumerate(data['movieId'].unique())}
n_users = len(user_ids)
n_movies = len(movie_ids)

data['userId'] = data['userId'].apply(lambda x: user_ids[x])
data['movieId'] = data['movieId'].apply(lambda x: movie_ids[x])
print(data.head(5))

# 2. Train-test split
train_data = data.sample(frac=0.8, random_state=123)
test_data = data.drop(train_data.index)

# Define Dataset
class RatingDataset(Dataset):
    def __init__(self, data):
        self.user_ids = data['userId'].values
        self.movie_ids = data['movieId'].values
        self.ratings = data['rating'].values
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]
        return user_id, movie_id, rating

train_dataset = RatingDataset(train_data)
test_dataset = RatingDataset(test_data)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 3. Define Model: Hybrid Matrix Factorization + Neural Collaborative Filtering (NCF)
class HybridRecSys(nn.Module):
    def __init__(self, n_users, n_movies, embedding_size=32, hidden_size=[64, 32, 16]):
        super(HybridRecSys, self).__init__()
        
        # Embeddings for MF
        self.user_embedding_mf = nn.Embedding(n_users, embedding_size)
        self.movie_embedding_mf = nn.Embedding(n_movies, embedding_size)
        
        # Embeddings for MLP part of NCF
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_size)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_size)
        
        # MLP layers
        layers = []
        input_size = embedding_size * 2  # User and movie embeddings concatenated
        for hidden in hidden_size:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            input_size = hidden
        self.mlp_layers = nn.Sequential(*layers)
        
        # Final layer combines MF and MLP for prediction
        # Adjusting the input size of the output layer to match mf_output + mlp_output
        self.output_layer = nn.Linear(hidden_size[-1] + 1, 1)
        
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, user_ids, movie_ids):
        # MF branch
        user_embed_mf = self.user_embedding_mf(user_ids)
        movie_embed_mf = self.movie_embedding_mf(movie_ids)
        mf_output = torch.mul(user_embed_mf, movie_embed_mf).sum(dim=1)  # Dot product
        
        # MLP branch
        user_embed_mlp = self.user_embedding_mlp(user_ids)
        movie_embed_mlp = self.movie_embedding_mlp(movie_ids)
        mlp_input = torch.cat([user_embed_mlp, movie_embed_mlp], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Final output combines both branches
        final_output = torch.cat([mf_output.unsqueeze(1), mlp_output], dim=1)
        final_output = self.output_layer(final_output)
        
        return final_output.squeeze(1)  # Remove extra dimension for final output


# Initialize model and optimizer
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model = HybridRecSys(n_users, n_movies, embedding_size=32, hidden_size=[128, 64, 32])
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 4. Train the Model
def train_model(train_loader, optimizer, model, criterion, num_epochs=5):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for user_id, movie_id, rating in tqdm(train_loader):
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)
            rating = rating.to(device, dtype=torch.float32) # Convert to float for MSE
            
            optimizer.zero_grad()
            outputs = model(user_id, movie_id)
            loss = criterion(outputs, rating)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(user_id)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'models/movie_recommendation_model_{epoch + 1}.pth')
            torch.save(model, 'models/movie_recommendation_model_complete.pth')

# Train the model
train_model(train_loader, optimizer, model, criterion, num_epochs=500)

# 6. Save the model
torch.save(model.state_dict(), 'models/movie_recommendation_model.pth')
torch.save(model, 'models/movie_recommendation_model_complete.pth')
print("Model saved!")


# 5. Evaluate the Model
def evaluate_model(test_loader, model):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for user_id, movie_id, rating in test_loader:
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)
            rating = rating.to(device, dtype=torch.float32)
            
            outputs = model(user_id, movie_id)
            preds.append(outputs.cpu().numpy())
            actuals.append(rating.cpu().numpy())
    
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    
    mse = mean_squared_error(actuals, preds)
    print(f"Test MSE: {mse:.4f}")
    return mse

evaluate_model(test_loader, model)