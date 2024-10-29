# DeepMovieRecommender 

## Running the Interface

```bash
cd interface/
python3 manage.py runserver
```
Append /search/ to the server url

## Repo Organization
`interface`: web interface

`models`: model training using combinations of neural network and collaborative filtering techniques

`ncf.py`: building the deep learning model

`test_ncf.ipynb`: testing the deep learning model

`test.py`: building the collaborative filtering model

`test.ipynb`: testing the collaborative filtering model

## Dataset
https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?resource=download
