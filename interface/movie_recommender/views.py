import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from django.shortcuts import render
from .models import MovieRating
from django.utils.html import format_html
from asgiref.sync import sync_to_async
from movie_recommender.rec_module import get_recs_for_user, group_and_output_explanations, output_image
from django.http import HttpResponse
from thefuzz import process
import time
import numpy as np

# Define paths for cache files
MOVIES_FILE = '../data/movies.csv'
RATINGS_FILE = '../data/ratings.csv'
MOVIES_RATINGS_FILE = '../cache/movies_and_ratings.pkl'
MOVIE_GENRES_FILE = '../cache/movie_genres.pkl'
MODEL_FILE = '../models/ncf.pth'
N_USERS_FILE = '../cache/n_users.pkl'
N_MOVIES_FILE = '../cache/n_movies.pkl'
MOVIE_IDS_FILE = '../cache/movie_ids.pkl'
N_GENRES_FILE = '../cache/n_genres.pkl'

# Initialize global variables
movies = None
movies_and_ratings = None
movie_genres = None
n_movies = None
n_users = None
movie_ids = None
n_genres = None
model = None

class EnhancedRecommendationModel(nn.Module):
    def __init__(self, n_users, n_movies, n_genres, n_factors=50):
        super(EnhancedRecommendationModel, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.genre_factors = nn.Embedding(n_genres, n_factors)
        self.fc1 = nn.Linear(n_factors * 2 + n_factors, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, user, movie, genres):
        user_embedding = self.user_factors(user)
        movie_embedding = self.movie_factors(movie)
        genre_embedding = self.genre_factors(genres)
        x = torch.cat([user_embedding, movie_embedding, genre_embedding], dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

def load_cached_data(just_movies = False):
    global movies_and_ratings, movie_genres, n_users, n_movies, movie_ids, n_genres, model, movies


    # Load movies_and_ratings DataFrame if exists
    print("Loading movie ratings file")
    if os.path.exists(MOVIES_RATINGS_FILE):
        with open(MOVIES_RATINGS_FILE, 'rb') as f:
            movies_and_ratings = pickle.load(f)
        with open(MOVIE_IDS_FILE, 'rb') as f:
            movie_ids = pickle.load(f)
    else:
        # Load from CSV and process
        movies_df = pd.read_csv(MOVIES_FILE)
        ratings_df = pd.read_csv(RATINGS_FILE)
        user_ids = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
        ratings_df['userId'] = ratings_df['userId'].apply(lambda x: user_ids[x])
        movies_and_ratings = pd.merge(movies_df, ratings_df, on="movieId")
        movie_ids = {id: i for i, id in enumerate(sorted(movies_and_ratings['movieId'].unique()))}
        movies_and_ratings['movieId'] = movies_and_ratings['movieId'].apply(lambda x: movie_ids[x])
        
        # Save processed data for future use
        with open(MOVIES_RATINGS_FILE, 'wb') as f:
            pickle.dump(movies_and_ratings, f)
        with open(MOVIE_IDS_FILE, 'wb') as f:
            pickle.dump(movie_ids, f)
        with open(N_USERS_FILE, 'wb') as f:
            pickle.dump(len(user_ids), f)
        with open(N_MOVIES_FILE, 'wb') as f:
            pickle.dump(len(movie_ids), f)

    if movies is None:
        movies = pd.read_csv('../data/movies.csv')
        ratings = pd.read_csv('../data/ratings.csv')
        movies = movies[movies['movieId'].isin(ratings['movieId'])]
        movies['movieId'] = movies['movieId'].apply(lambda x: movie_ids[x])


        movies = movies_and_ratings.groupby('movieId', as_index=False).first()


    if just_movies:
        return

    # Load genres if exists
    print("Loading movie genres file")
    if os.path.exists(MOVIE_GENRES_FILE):
        with open(MOVIE_GENRES_FILE, 'rb') as f:
            movie_genres = pickle.load(f)
    else:
        # Compute genres from movies_and_ratings
        movies_and_ratings['genres'] = movies_and_ratings['genres'].str.split('|')
        genres_set = set(g for sublist in movies_and_ratings['genres'] for g in sublist)
        n_genres = len(genres_set)
        for genre in genres_set:
            movies_and_ratings[genre] = movies_and_ratings['genres'].apply(lambda x: int(genre in x))
        movie_genres = movies_and_ratings[['movieId'] + list(genres_set)].drop_duplicates().set_index('movieId')
        
        # Save processed data for future use
        with open(MOVIE_GENRES_FILE, 'wb') as f:
            pickle.dump(movie_genres, f)
        with open(N_GENRES_FILE, 'wb') as f:
            pickle.dump(n_genres, f)

    print("Loading model")
    # Load model if exists
    if os.path.exists(MODEL_FILE):
        if model is None:
            with open(N_USERS_FILE, 'rb') as f:
                n_users = pickle.load(f)
            with open(N_MOVIES_FILE, 'rb') as f:
                n_movies = pickle.load(f)
            with open(N_GENRES_FILE, 'rb') as f:
                n_genres = pickle.load(f)
            model = EnhancedRecommendationModel(n_users, n_movies, n_genres).to('cpu')
            state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)

def search_view(request):
    # all_people = Person.objects.all()
    # context = {'count': all_people.count()}
    print("Hello")
    load_cached_data()
    return render(request, 'search.html', None)

def set_rating(request, movie_id):
    if request.method == 'POST':
        form_data = request.POST.dict()
        print(form_data)
        rating = form_data.get("number")

        MovieRating(id=movie_id, rating=rating).save()

        return render(request, 'rating.html', {"rating": rating})

def search_results_view(request):
    query = request.GET.get('search', '')
    print(f'{query = }')
    global movies

    from thefuzz import process 

    print("Loading movies")
    if movies is None:
        load_cached_data(True)


    print("Movies loaded")
    matches = process.extract(query, movies["title"], limit=10)

    formatted_matches = [
        {"name": name, "score": distance, "id": index}
        for (name, distance, index) in matches
    ]

    ratings = MovieRating.objects.filter(id__in=map(lambda x: x['id'], formatted_matches))

    rating_map = {}

    for rating in ratings:
        rating_map[rating.id] = rating.rating

    sorted(formatted_matches, key=lambda x: x['score'])

    for match in formatted_matches:
        match["rating"] = rating_map.get(match['id'], None)

    context = {'movies': formatted_matches}
    return render(request, 'search_results.html', context)


def get_recommendations(request):
    ratings = MovieRating.objects.all()

    rated_movie_ids = []
    movie_ratings = []
    for rating in ratings:
        rated_movie_ids.append(rating.id)
        movie_ratings.append(rating.rating)

    # Calculate recommendations
    global movie_genres, model, movie_ids, movies
    print("Loading movie ratings, movie genres, and model")
    if movie_genres is None or model is None or movie_ids is None or movies is None:
        load_cached_data()
    print("Everything loaded")
    print("Getting recommendations for user")
    if len(rated_movie_ids) == 1:
        rated_movie_ids = [rated_movie_ids[0], rated_movie_ids[0]]
        movie_ratings = [movie_ratings[0], movie_ratings[0]]
    new_user_ratings_df = pd.DataFrame([rated_movie_ids, movie_ratings])
    print(new_user_ratings_df)

    recommended_movie_ids, movie_embedding_matrix = get_recs_for_user(new_user_ratings_df, movie_genres, model)
    print(recommended_movie_ids)

    user_rating_ids = np.unique(new_user_ratings_df.T[0].values)

    explanations = group_and_output_explanations(recommended_movie_ids, [], user_rating_ids, movies)["Hybrid Recommendations"]
    ######################################

    def get_explaination(movie_title):
        for title, explaination in explanations:
            if movie_title == title:
                return explaination
        return "No explanation available"
    


    formatted_recommended_movies = [{"name": movie.title, "explaination": get_explaination(movie.title)} for (_, movie) in movies[movies["movieId"].isin(recommended_movie_ids)].iterrows()]

    image = output_image(recommended_movies_hybrid_movie_ids=recommended_movie_ids, new_user_movie_ids=user_rating_ids, movie_embedding_matrix=movie_embedding_matrix, movies=movies)

    return render(request, "recommendations.html", {"movies": formatted_recommended_movies, "image": image})


def delete_all_ratings(request):
    MovieRating.objects.all().delete()

    response = HttpResponse()
    response.headers["HX-Refresh"] = "true"
    return response

def highlight_matched_text(text, query):
    """
    Inserts html around the matched text.
    """
    start = text.lower().find(query.lower())
    if start == -1:
        return text
    end = start + len(query)
    highlighted = format_html('<span class="highlight">{}</span>', text[start:end])
    return format_html('{}{}{}', text[:start], highlighted, text[end:])

