# urls.py in sim
from django.urls import path
from . import views

urlpatterns = [
    path('', views.search_view, name='search_view'),
    path('results', views.search_results_view, name='search_results_view'),
    path('submit-number/<int:movie_id>', views.set_rating, name='set_rating'),
]
