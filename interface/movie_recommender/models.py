# models.py in sim
from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

class Person(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return self.name
    

class MovieRating(models.Model):
    id = models.IntegerField(primary_key=True)
    rating = models.IntegerField(validators=[
            MaxValueValidator(10),
            MinValueValidator(0)
        ])