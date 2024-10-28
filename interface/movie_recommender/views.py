
from django.shortcuts import render
from .models import MovieRating
from django.utils.html import format_html
import sys
import pandas as pd
import csv
import torch
from asgiref.sync import sync_to_async
import torch.nn as nn
from movie_recommender.rec_module import get_recs_for_user

from django.http import HttpResponse
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
    


def search_view(request):
    # all_people = Person.objects.all()
    # context = {'count': all_people.count()}
    print("Hello")
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
    global movies, movies_and_ratings, n_users, n_movies, movie_ids

    # Should find a way to do this asynchronously if possible
    if movies_and_ratings is None:
        # Load movies
        movies_df = pd.read_csv('../data/movies.csv')
        ratings_df = pd.read_csv('../data/ratings.csv')
        # Convert user and item IDs to integers (index-based)
        user_ids = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
        n_users = len(user_ids)
        ratings_df['userId'] = ratings_df['userId'].apply(lambda x: user_ids[x])
        movies_and_ratings = pd.merge(movies_df, ratings_df, on="movieId")
        movie_ids = {id: i for i, id in enumerate(sorted(movies_and_ratings['movieId'].unique()))}
        movies_and_ratings['movieId'] = movies_and_ratings['movieId'].apply(lambda x: movie_ids[x])
        n_movies = len(movie_ids)
    
    if movies is None:
        movies = pd.read_csv('../data/movies.csv')
    

    from thefuzz import process 

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
    global movies_and_ratings, movie_genres, n_users, n_movies, n_genres, model, movie_ids, movies

    if movies_and_ratings is None or n_movies is None:
        movies_df = pd.read_csv('../data/movies.csv')
        ratings_df = pd.read_csv('../data/ratings.csv')
        user_ids = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
        n_users = len(user_ids)
        ratings_df['userId'] = ratings_df['userId'].apply(lambda x: user_ids[x])
        movies_and_ratings = pd.merge(movies_df, ratings_df, on="movieId")
        movie_ids = {id: i for i, id in enumerate(sorted(movies_and_ratings['movieId'].unique()))}
        movies_and_ratings['movieId'] = movies_and_ratings['movieId'].apply(lambda x: movie_ids[x])
        n_movies = len(movie_ids)

    if movie_genres is None or n_genres is None:
        movies_and_ratings['genres'] = movies_and_ratings['genres'].str.split('|')
        genres_set = set(g for sublist in movies_and_ratings['genres'] for g in sublist)
        n_genres = len(genres_set)
        for genre in genres_set:
            movies_and_ratings[genre] = movies_and_ratings['genres'].apply(lambda x: int(genre in x))
        movie_genres = movies_and_ratings[['movieId'] + list(genres_set)].drop_duplicates().set_index('movieId')
        
    if model is None:
        model = EnhancedRecommendationModel(n_users, n_movies, n_genres).to('cpu')
        state_dict = torch.load('../models/ncf.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    new_user_ratings_df = pd.DataFrame([rated_movie_ids, movie_ratings])
    new_user_ratings_df[0] = new_user_ratings_df[0].apply(lambda x: movie_ids[x])
    
    if movies is None:
        movies = pd.read_csv('../data/movies.csv')
        ratings = pd.read_csv('../data/ratings.csv')
        movies = movies[movies['movieId'].isin(ratings['movieId'])]
        movies['movieId'] = movies['movieId'].apply(lambda x: movie_ids[x])
    
    recommended_movie_ids = get_recs_for_user(new_user_ratings_df, movie_genres, model)
    print(recommended_movie_ids)
    ######################################


    formatted_recommended_movies = [{"name": movie.title} for (_, movie) in movies[movies["movieId"].isin(recommended_movie_ids)].iterrows()]

    return render(request, "recommendations.html", {"movies": formatted_recommended_movies})


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

