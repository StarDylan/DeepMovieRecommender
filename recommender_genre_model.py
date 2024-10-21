import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import csv

# 1. Load the datasets
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv', low_memory=False)

# 1.1 Encode genres with MultiLabelBinarizer (MLB)
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))  # Split genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies['genres'])  # Convert genres to binary encoding
genre_columns = mlb.classes_

# Add the genre columns to the movies DataFrame
genres_df = pd.DataFrame(genres_encoded, columns=genre_columns)
movies = pd.concat([movies, genres_df], axis=1)

# Create a dictionary mapping movie IDs to their titles
movie_id_to_title = {row['movieId']: row['title'] for _, row in movies.iterrows()}

# Merge the datasets
data = pd.merge(movies, ratings, on='movieId')

# Convert user and movie IDs to integer indices (index-based)
user_ids = {id: i for i, id in enumerate(data['userId'].unique())}
movie_ids = {id: i for i, id in enumerate(data['movieId'].unique())}
n_users = len(user_ids)
n_movies = len(movie_ids)
n_genres = len(genre_columns)  # Number of genres

data['userId'] = data['userId'].apply(lambda x: user_ids[x])
data['movieId'] = data['movieId'].apply(lambda x: movie_ids[x])

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)

# 2. Define Dataset class
class RatingDataset(Dataset):
    def __init__(self, data, genre_columns):
        self.user_ids = data['userId'].values
        self.movie_ids = data['movieId'].values
        self.ratings = data['rating'].values
        self.genres = data[genre_columns].values
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]
        genres = torch.tensor(self.genres[idx], dtype=torch.float32)
        return user_id, movie_id, genres, rating

# Create train and test datasets
train_dataset = RatingDataset(train_data, genre_columns)
test_dataset = RatingDataset(test_data, genre_columns)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # Larger batch size for efficiency
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 3. Define Model: Hybrid Matrix Factorization + NCF + Genre Embeddings
class HybridRecSys(nn.Module):
    def __init__(self, n_users, n_movies, n_genres, embedding_size=32, hidden_size=[64, 32]):
        super(HybridRecSys, self).__init__()
        
        # Embeddings for MF
        self.user_embedding_mf = nn.Embedding(n_users, embedding_size)
        self.movie_embedding_mf = nn.Embedding(n_movies, embedding_size)
        
        # Embeddings for MLP part of NCF
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_size)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_size)
        
        # Genre embedding layer (input: one-hot vector of genres)
        self.genre_embedding = nn.Linear(n_genres, embedding_size)
        
        # MLP layers
        layers = []
        input_size = embedding_size * 3  # User, movie, and genre embeddings concatenated
        for hidden in hidden_size:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            input_size = hidden
        self.mlp_layers = nn.Sequential(*layers)
        
        # Final layer combines MF and MLP for prediction
        self.output_layer = nn.Linear(hidden_size[-1] + 1, 1)
    
    def forward(self, user_ids, movie_ids, genres):
        # MF branch
        user_embed_mf = self.user_embedding_mf(user_ids)
        movie_embed_mf = self.movie_embedding_mf(movie_ids)
        mf_output = torch.mul(user_embed_mf, movie_embed_mf).sum(dim=1)  # Dot product
        
        # MLP branch
        user_embed_mlp = self.user_embedding_mlp(user_ids)
        movie_embed_mlp = self.movie_embedding_mlp(movie_ids)
        genre_embed_mlp = self.genre_embedding(genres)
        mlp_input = torch.cat([user_embed_mlp, movie_embed_mlp, genre_embed_mlp], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # Final output combines both branches
        final_output = torch.cat([mf_output.unsqueeze(1), mlp_output], dim=1)
        final_output = self.output_layer(final_output)
        
        return final_output.squeeze(1)  # Remove extra dimension for final output

# 4. Initialize model, criterion, optimizer
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
model = HybridRecSys(n_users, n_movies, n_genres, embedding_size=32, hidden_size=[128, 64])
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# 5. Train the model
def train_model(train_loader, optimizer, model, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for user_id, movie_id, genres, rating in tqdm(train_loader):
            user_id, movie_id, genres, rating = user_id.to(device), movie_id.to(device), genres.to(device), rating.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(user_id, movie_id, genres)
            loss = criterion(outputs, rating)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(user_id)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Train the model
train_model(train_loader, optimizer, model, criterion, num_epochs=5)

# 6. Save the model
torch.save(model.state_dict(), 'movie_recommendation_model.pth')
print("Model saved!")

# 7. Evaluate the model
def evaluate_model(test_loader, model):
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for user_id, movie_id, genres, rating in tqdm(test_loader):
            user_id, movie_id, genres, rating = user_id.to(device), movie_id.to(device), genres.to(device), rating.to(device).float()
            
            outputs = model(user_id, movie_id, genres)
            preds.append(outputs.cpu().numpy())
            actuals.append(rating.cpu().numpy())
    
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"Test RMSE: {rmse:.4f}")
    return rmse

# Evaluate the model
evaluate_model(test_loader, model)