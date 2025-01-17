import streamlit as st
import pickle
import pandas as pd

def fetch_poster(movie):
    response = request.get()
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies=[]
    for i in movie_list:
        movie_id = i[0]
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies
movies_dict = pickle.load((open('movies.pkl', 'rb')))
movies = pd.DataFrame(movies_dict)

similarity=pickle.load((open('similarity.pkl', 'rb')))
st.title("Movie Recommender System")
opt = st.selectbox('How would you like to be contacted ?', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(opt)
    for i in recommendations:
        st.write(i)