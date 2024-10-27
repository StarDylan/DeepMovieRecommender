
from django.shortcuts import render
from .models import MovieRating
from django.utils.html import format_html
import sys
import pandas as pd

from django.http import HttpResponse
movies = None

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
    global movies

    if movies is None:
        # Load movies
        movies = pd.read_csv('../data/movies.csv')
        # Convert user and item IDs to integers (index-based)
        movie_ids = {id: i for i, id in enumerate(movies['movieId'].unique())}

        movies['movieId'] = movies['movieId'].apply(lambda x: movie_ids[x])


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
    print(rated_movie_ids, movie_ratings)
    ######################################

    recommended_movie_ids = [0, 1, 2]
    movies = MovieRating.objects.filter(id__in=recommended_movie_ids)

    return render(request, "recommendations.html", {"movies": movies})


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

