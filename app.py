import streamlit as st
import pickle
import pandas as pd
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Fetch movie poster from TMDB API
def fetch_poster(movie_id):

    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=095c918ec39cdc2d25b163f9b0757d08&language=en-US'.format(movie_id)
    )

    data = response.json()

    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


# Recommendation function
def recommend(movie):

    # Find movie index
    movie_index = movies[movies['title'] == movie].index[0]

    # Get similarity scores
    distances = similarity[movie_index]

    # Get top 5 similar movies
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:

        # Get movie id
        movie_id = movies.iloc[i[0]]['id']

        # Append movie title
        recommended_movies.append(movies.iloc[i[0]].title)

        # Append movie poster
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters


# Load movie dictionary
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))

# Convert dictionary into dataframe
movies = pd.DataFrame(movies_dict)


# Create vectors
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()


# Calculate similarity
similarity = cosine_similarity(vectors)


# Streamlit title
st.title('Movie Recommender System')


# Dropdown menu
selected_movie_name = st.selectbox(
    'Search for a movie',
    movies['title'].values
)


# Recommend button
if st.button('Recommend'):

    names, posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])