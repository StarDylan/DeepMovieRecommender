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
    new_user_movie_ids = new_user_ratings_df[0]
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

def explain_content_based_recommendations(recommended_movies, user_favorite_genres, movies):
    explanations = []
    for movie_id in recommended_movies:
        title = movies[movies["movieId"] == movie_id].title.values[0]
        genres = movies[movies["movieId"] == movie_id]["genres"].tolist()
        common_genres = set(genres).intersection(set(user_favorite_genres))
        explanation = f"'{title}' is recommended through content-based recommendation because it belongs to your favorite genres: {', '.join(common_genres)}."
        explanations.append((title, explanation))
    return explanations

def explain_latent_factor_recommendations(recommended_movies, user_ratings_df, movies):
    print(f"User Ratings df T[0]: {user_ratings_df.T[0]}")
    explanations = []
    for movie_id in recommended_movies:
        title = movies[movies["movieId"] == movie_id].title.values[0]
        similar_movies = user_ratings_df.T[0].apply(lambda x: movies[movies["movieId"] == x].title.values[0]).tolist()
        explanation = f"'{title}' is recommended through latent factor recommendation because it is similar to movies you rated highly: {', '.join(similar_movies)}."
        explanations.append((title, explanation))
    return explanations

def explain_hybrid_recommendations(hybrid_recommendations, user_favorite_genres, user_ratings_df, movies):
    content_explanations = explain_content_based_recommendations(hybrid_recommendations, user_favorite_genres, movies)
    latent_explanations = explain_latent_factor_recommendations(hybrid_recommendations.tolist(), user_ratings_df, movies)
    combined_explanations = []
    for content_exp, latent_exp in zip(content_explanations, latent_explanations):
        combined_explanations.append((content_exp[0], f"{content_exp[1]} Additionally, {latent_exp[1]}"))
    return combined_explanations

def group_and_output_explanations(hybrid_recommendations, user_favorite_genres, user_ratings_df, movies):
    content_explanations = explain_content_based_recommendations(hybrid_recommendations, user_favorite_genres, movies)
    latent_explanations = explain_latent_factor_recommendations(hybrid_recommendations.tolist(), user_ratings_df, movies)
    hybrid_explanations = explain_hybrid_recommendations(hybrid_recommendations, user_favorite_genres, user_ratings_df, movies)
    
    grouped_explanations = {
        "Content-Based Recommendations": content_explanations,
        "Latent Factor Recommendations": latent_explanations,
        "Hybrid Recommendations": hybrid_explanations
    }
    
    for group, explanations in grouped_explanations.items():
        print(f"\n{group}:\n")
        for title, explanation in explanations:
            print(explanation)

    return grouped_explanations
