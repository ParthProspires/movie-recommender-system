import streamlit as st
import pickle
import pandas as pd
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1 {
    text-align: center;
    color: #E50914;
    font-size: 50px;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 200px;
    font-size: 20px;
    border: none;
}

.stButton>button:hover {
    background-color: #ff1e28;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ---------------- FETCH POSTER ---------------- #

def fetch_poster(movie_id):

    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=095c918ec39cdc2d25b163f9b0757d08&language=en-US'
    )

    data = response.json()

    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]]['id']

        recommended_movies.append(movies.iloc[i[0]].title)

        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters


# ---------------- LOAD DATA ---------------- #

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))

movies = pd.DataFrame(movies_dict)


# ---------------- CREATE SIMILARITY ---------------- #

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


# ---------------- TITLE ---------------- #

st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

st.write("### Find movies similar to your favorite one 🍿")


# ---------------- SELECT MOVIE ---------------- #

selected_movie_name = st.selectbox(
    "Type or select a movie",
    movies['title'].values
)


# ---------------- BUTTON ---------------- #

if st.button('Show Recommendation'):

    with st.spinner('Finding best movies for you...'):

        names, posters = recommend(selected_movie_name)

        st.write("## Recommended Movies ❤️")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(posters[0])
            st.caption(names[0])

        with col2:
            st.image(posters[1])
            st.caption(names[1])

        with col3:
            st.image(posters[2])
            st.caption(names[2])

        with col4:
            st.image(posters[3])
            st.caption(names[3])

        with col5:
            st.image(posters[4])
            st.caption(names[4])


# ---------------- FOOTER ---------------- #

st.markdown("""
<hr>
<center>
Made with ❤️ using Streamlit
</center>
""", unsafe_allow_html=True)