import pandas as pd
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity


def get_recs_for_user(new_user_ratings_df, movie_genres, model):
    # Content-based
    user_genre_profile = np.dot(new_user_ratings_df[1], movie_genres.loc[new_user_ratings_df[0]])
    user_genre_profile = user_genre_profile.reshape(1, -1)
    genre_similarities = cosine_similarity(user_genre_profile, movie_genres.values)[0]

    # Latent
    movie_embedding_matrix = model.movie_factors.weight.data.cpu().numpy()
    new_user_movie_ids = new_user_ratings_df[0].values
    user_profile = movie_embedding_matrix[new_user_movie_ids].mean(axis=0)
    latent_similarities = cosine_similarity([user_profile], movie_embedding_matrix)[0]

    # Combined the two
    content_weight = 0.5
    latent_weight = 0.5
    genre_similarity_scores = genre_similarities / genre_similarities.max()
    latent_similarity_scores = pd.Series(latent_similarities).rank(pct=True)
    hybrid_scores = content_weight * genre_similarity_scores + latent_weight * latent_similarity_scores
    recommended_movies_hybrid_movie_ids = hybrid_scores.sort_values(ascending=False).head(10).index

    return recommended_movies_hybrid_movie_ids