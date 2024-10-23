import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import csv
from tqdm import tqdm

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

# Fill missing values with 0
user_movie_matrix.fillna(0, inplace=True)

# Split into training and test sets
train_data, test_data = train_test_split(df[['userId', 'movieId', 'rating']], test_size=0.2)

# Convert to tensors for PyTorch
train_tensor = torch.tensor(train_data.values, dtype=torch.float32).to('mps')
test_tensor = torch.tensor(test_data.values, dtype=torch.float32).to('mps')

# Get the number of users and movies
n_users = df['userId'].nunique()
n_movies = df['movieId'].nunique()

# Define the Matrix Factorization model
class MF(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=20):
        super(MF, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)  # User latent factors
        self.movie_factors = nn.Embedding(n_movies, n_factors)  # Movie latent factors
        self.user_biases = nn.Embedding(n_users, 1)  # User biases
        self.movie_biases = nn.Embedding(n_movies, 1)  # Movie biases

    def forward(self, user, movie):
        # Matrix factorization: dot product of user and movie latent factors + bias
        pred = (self.user_factors(user) * self.movie_factors(movie)).sum(1)
        pred += self.user_biases(user).squeeze() + self.movie_biases(movie).squeeze()
        return pred

# Initialize model, loss function, and optimizer
model = MF(n_users, n_movies).to('mps')
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Function to evaluate RMSE
def rmse(predictions, targets):
    return np.sqrt(mean_squared_error(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy()))

# Train the model
n_epochs = 10
batch_size = 1024

for epoch in range(n_epochs):
    model.train()
    losses = []
    for i in tqdm(range(0, len(train_tensor), batch_size)):
        batch = train_tensor[i:i+batch_size]
        users = batch[:, 0].long()
        movies = batch[:, 1].long()
        ratings = batch[:, 2]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(users, movies)
        loss = loss_fn(preds, ratings)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Print loss at the end of each epoch
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {np.mean(losses)}')

torch.save(model.state_dict(), 'models/new_movie_recommendation_model.pth')
torch.save(model, 'models/new_movie_recommendation_model_complete.pth')

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    users_test = test_tensor[:, 0].long()
    movies_test = test_tensor[:, 1].long()
    ratings_test = test_tensor[:, 2]
    
    preds_test = model(users_test, movies_test)
    test_rmse = rmse(preds_test, ratings_test)
    print(f'Test RMSE: {test_rmse}')

# Example input for a new user
new_user_ratings = {
    1: 5.0,  # MovieId 1, rating 5.0
    2: 3.0,  # MovieId 2, rating 3.0
    50: 4.5  # MovieId 50, rating 4.5
}

def recommend_genre_based(new_user_ratings, df, top_n=10):
    # Create a DataFrame for the movies the new user has rated
    rated_movies = pd.DataFrame(new_user_ratings.items(), columns=['movieId', 'rating'])
    
    # Merge the rated movies with the original dataframe to get genres
    rated_movies = pd.merge(rated_movies, df[['movieId', 'title'] + list(genres_set)], on='movieId')
    
    # Get the genre profile by weighting the genres of the movies the user liked
    genre_profile = rated_movies[genres_set].T.dot(rated_movies['rating'])
    
    # Normalize the genre profile
    genre_profile /= genre_profile.sum()
    
    # Calculate the similarity of other movies to this genre profile
    all_movies_genres = df[['movieId', 'title'] + list(genres_set)]
    all_movies_genres['similarity'] = all_movies_genres[genres_set].dot(genre_profile)
    
    # Exclude movies that the user has already rated
    all_movies_genres = all_movies_genres[~all_movies_genres['movieId'].isin(new_user_ratings.keys())]
    
    # Recommend the top N movies with the highest similarity scores
    recommendations = all_movies_genres[['movieId', 'title', 'similarity']].sort_values(by='similarity', ascending=False).head(top_n)
    return recommendations

# Example usage:
recommendations = recommend_genre_based(new_user_ratings, df)
print(recommendations)