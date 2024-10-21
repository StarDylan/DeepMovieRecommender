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

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv', low_memory=False)

movie_id_to_title = {}
with open('data/movies.csv', 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        movie_id = int(row[0])
        title = row[1]
        movie_id_to_title[movie_id] = title

data = pd.merge(movies, ratings, on='movieId')

user_ids = {id: i for i, id in enumerate(data['userId'].unique())}
movie_ids = {id: i for i, id in enumerate(data['movieId'].unique())}
n_users = len(user_ids)
n_movies = len(movie_ids)

data['userId'] = data['userId'].apply(lambda x: user_ids[x])
data['movieId'] = data['movieId'].apply(lambda x: movie_ids[x])


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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)




class HybridRecSys(nn.Module):
    def __init__(self, n_users, n_movies, embedding_size=32, hidden_size=[64, 32, 16]):
        super(HybridRecSys, self).__init__()
        
        self.user_embedding_mf = nn.Embedding(n_users, embedding_size)
        self.movie_embedding_mf = nn.Embedding(n_movies, embedding_size)
        
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_size)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_size)
        
        layers = []
        input_size = embedding_size * 2
        for hidden in hidden_size:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            input_size = hidden
        self.mlp_layers = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(hidden_size[-1] + 1, 1)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, user_ids, movie_ids):
        user_embed_mf = self.user_embedding_mf(user_ids)
        movie_embed_mf = self.movie_embedding_mf(movie_ids)
        mf_output = torch.mul(user_embed_mf, movie_embed_mf).sum(dim=1)

        user_embed_mlp = self.user_embedding_mlp(user_ids)
        movie_embed_mlp = self.movie_embedding_mlp(movie_ids)
        mlp_input = torch.cat([user_embed_mlp, movie_embed_mlp], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        
        final_output = torch.cat([mf_output.unsqueeze(1), mlp_output], dim=1)
        final_output = self.output_layer(final_output)
        
        return final_output.squeeze(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

import os
for model in os.listdir('models'):
    if 'complete' in model:
        continue

    loaded_model = HybridRecSys(n_users, n_movies, embedding_size=32, hidden_size=[128, 64, 32])
    loaded_model.load_state_dict(torch.load(os.path.join('models', model)))
    loaded_model.to(device)
    loaded_model.eval()
    print("Model loaded!")


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

    evaluate_model(test_loader, loaded_model)


    def recommend_top_n(user_id, model, top_n=10):
        model.eval()
        user_embedding = model.user_embedding_mf(torch.tensor([user_id]).to(device))
        all_movie_embeddings = model.movie_embedding_mf.weight.data 

        # Dot product similarity
        scores = torch.matmul(user_embedding, all_movie_embeddings.T).squeeze(0)
        top_movie_ids = torch.topk(scores, top_n).indices.cpu().numpy()

        # Get corresponding movie titles
        top_movie_titles = [movie_id_to_title[movie_ids_inv[movie_id]] for movie_id in top_movie_ids]
        return top_movie_titles

movie_ids_inv = {v: k for k, v in movie_ids.items()}  # Reverse movie_id mapping for recommendation
recommendations = recommend_top_n(user_id=12, model=loaded_model, top_n=10)
print("Top 10 recommendations for user 12:", recommendations)
