import pandas as pd
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from matplotlib.figure import Figure

TSNE_FILE = '../cache/tsne.pkl'

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
    recommended_movies_hybrid_movie_ids = hybrid_scores.sort_values(ascending=False).head(5).index

    return recommended_movies_hybrid_movie_ids, movie_embedding_matrix

def explain_content_based_recommendations(recommended_movies, user_favorite_genres, movies_and_ratings):
    explanations = []
    for movie_id in recommended_movies:
        title = movies_and_ratings[movies_and_ratings["movieId"] == movie_id].title.values[0]
        genres = movies_and_ratings[movies_and_ratings["movieId"] == movie_id]["genres"].tolist()
        common_genres = set(genres).intersection(set(user_favorite_genres))
        explanation = f"'{title}' is recommended through content-based recommendation because it belongs to your favorite genres: {', '.join(common_genres)}."
        explanations.append((title, explanation))
    return explanations

def explain_latent_factor_recommendations(recommended_movies, user_ratings_df, movies_and_ratings):
    explanations = []
    for movie_id in recommended_movies:
        title = movies_and_ratings[movies_and_ratings["movieId"] == movie_id].title.values[0]
        similar_movies = list(map(lambda x: movies_and_ratings[movies_and_ratings["movieId"] == x].title.values[0], user_ratings_df))
        explanation = f"'{title}' is recommended through latent factor recommendation because it is similar to movies you rated highly: {', '.join(similar_movies)}."
        explanations.append((title, explanation))
    return explanations

def explain_hybrid_recommendations(hybrid_recommendations, user_favorite_genres, user_ratings_df, movies_and_ratings):
    content_explanations = explain_content_based_recommendations(hybrid_recommendations, user_favorite_genres, movies_and_ratings)
    latent_explanations = explain_latent_factor_recommendations(hybrid_recommendations.tolist(), user_ratings_df, movies_and_ratings)
    combined_explanations = []
    for content_exp, latent_exp in zip(content_explanations, latent_explanations):
        combined_explanations.append((content_exp[0], f"{content_exp[1]} Additionally, {latent_exp[1]}"))
    return combined_explanations

def group_and_output_explanations(hybrid_recommendations, user_favorite_genres, user_ratings_df, movies_and_ratings):
    content_explanations = explain_content_based_recommendations(hybrid_recommendations, user_favorite_genres, movies_and_ratings)
    latent_explanations = explain_latent_factor_recommendations(hybrid_recommendations.tolist(), user_ratings_df, movies_and_ratings)
    hybrid_explanations = explain_hybrid_recommendations(hybrid_recommendations, user_favorite_genres, user_ratings_df, movies_and_ratings)
    
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

embeddings_2d = None

def output_image(recommended_movies_hybrid_movie_ids, new_user_movie_ids, movie_embedding_matrix, movies):
    from sklearn.manifold import TSNE


    # Dimensionality reduction using TSNE
    global embeddings_2d

    if embeddings_2d is None:
        if os.path.exists(TSNE_FILE):
            with open(TSNE_FILE, 'rb') as f:
                embeddings_2d = pickle.load(f)
        else:
            # Calculate TSNE + save
            print(f"Fitting TSNE on {movie_embedding_matrix.shape[0]} movie embeddings...")
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(movie_embedding_matrix)
            
            # Save processed data for future use
            with open(TSNE_FILE, 'wb') as f:
                pickle.dump(embeddings_2d, f)
            
            print("Completed TSNE fitting + Saved to cache")



    # Visualization
    fig = Figure(figsize=(14, 10))
    ax = fig.subplots()

    # Plot all movie embeddings
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, label='All Movies')

    # min max of embedding dems
    print(f"Min: {embeddings_2d[:, 0].min()}, Max: {embeddings_2d[:, 0].max()}")
    print(f"Min: {embeddings_2d[:, 1].min()}, Max: {embeddings_2d[:, 1].max()}")



    # # Highlight recommended movies
    recommended_movie_ids = recommended_movies_hybrid_movie_ids
    recommended_embeddings = embeddings_2d[recommended_movie_ids]
    ax.scatter(recommended_embeddings[:, 0], recommended_embeddings[:, 1], color='blue', label='Recommended Movies')

    # Highlight user-rated movies
    user_rated_embeddings = embeddings_2d[new_user_movie_ids]
    print(f"User embeddings: {user_rated_embeddings}")
    ax.scatter(user_rated_embeddings[:, 0], user_rated_embeddings[:, 1], color='red', label='User Rated Movies')


    # # Annotate recommended movies
    for i, movie_id in enumerate(recommended_movie_ids):
        ax.annotate(movies[movies["movieId"] == movie_id].title.values[0], (recommended_embeddings[i, 0], recommended_embeddings[i, 1]), color='blue')
        
    # # Annotate user-rated movies
    for i, movie_id in enumerate(new_user_movie_ids):
        print(movie_id)
        print(f"Annotated: {movies[movies['movieId'] == movie_id].title.values[0]}, {(user_rated_embeddings[i, 0], user_rated_embeddings[i, 1])}")
        ax.annotate(movies[movies['movieId'] == movie_id].title.values[0], (user_rated_embeddings[i, 0], user_rated_embeddings[i, 1]), color='red')

    ax.set_title('2D Visualization of Movie Embeddings')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend()

    import io
    import base64
    my_stringIObytes = io.BytesIO()
    ax.figure.savefig(my_stringIObytes, transparent=True)
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

    return my_base64_jpgData