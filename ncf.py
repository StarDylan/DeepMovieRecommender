import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import csv
from tqdm import tqdm

# Load movies and ratings data
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
df = pd.merge(movies, ratings, on='movieId')

# Convert user and item IDs to integers (index-based)
user_ids = {id: i for i, id in enumerate(df['userId'].unique())}
movie_ids = {id: i for i, id in enumerate(df['movieId'].unique())}
n_users = len(user_ids)
n_movies = len(movie_ids)

df['userId'] = df['userId'].apply(lambda x: user_ids[x])
df['movieId'] = df['movieId'].apply(lambda x: movie_ids[x])

# One-hot encoding genres
df['genres'] = df['genres'].str.split('|')
genres_set = set(g for sublist in df['genres'] for g in sublist)
for genre in genres_set:
    df[genre] = df['genres'].apply(lambda x: int(genre in x))

# Create user-item interaction matrix
user_movie_matrix = df.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Split into training and test sets
train_data, test_data = train_test_split(df[['userId', 'movieId', 'rating']], test_size=0.2)

# Convert to tensors for PyTorch
train_tensor = torch.tensor(train_data.values, dtype=torch.float32).to('mps')
test_tensor = torch.tensor(test_data.values, dtype=torch.float32).to('mps')

# Define the Enhanced Recommendation Model
class EnhancedRecommendationModel(nn.Module):
    def __init__(self, n_users, n_movies, n_genres, n_factors=50):
        super(EnhancedRecommendationModel, self).__init__()
        # User and movie latent factors
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.genre_factors = nn.Embedding(n_genres, n_factors)  # Genre embeddings
        
        # Fully connected layers
        self.fc1 = nn.Linear(n_factors * 2 + n_factors, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, user, movie, genres):
        user_embedding = self.user_factors(user)
        movie_embedding = self.movie_factors(movie)
        genre_embedding = self.genre_factors(genres)

        # Concatenate user, movie, and genre embeddings
        x = torch.cat([user_embedding, movie_embedding, genre_embedding], dim=1)
        
        # Pass through fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

# Initialize model, loss function, and optimizer
n_genres = len(genres_set)  # Number of unique genres
model = EnhancedRecommendationModel(n_users, n_movies, n_genres).to('mps')
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to evaluate RMSE
def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy()))

# Train the model
n_epochs = 50
batch_size = 256

for epoch in range(n_epochs):
    model.train()
    losses = []
    for i in tqdm(range(0, len(train_tensor), batch_size)):
        batch = train_tensor[i:i+batch_size]
        users = batch[:, 0].long()
        movies = batch[:, 1].long()
        ratings = batch[:, 2]

        # Improve this later
        genres = torch.zeros_like(users).long() 

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(users, movies, genres)
        loss = loss_fn(preds.view(-1), ratings)  # Flatten predictions
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Print loss at the end of each epoch
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {np.mean(losses)}')

# Save the model
torch.save(model.state_dict(), 'models/enhanced_movie_recommendation_model.pth')

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    users_test = test_tensor[:, 0].long()
    movies_test = test_tensor[:, 1].long()
    ratings_test = test_tensor[:, 2]
    
    # Use the same dummy genre input
    genres_test = torch.zeros_like(users_test).long()
    
    preds_test = model(users_test, movies_test, genres_test)
    test_rmse = rmse(preds_test.view(-1), ratings_test)  # Flatten predictions
    print(f'Test RMSE: {test_rmse}')
