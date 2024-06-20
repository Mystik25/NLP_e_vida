pip install -r requerimento.txt

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re


# Função para obter reviews do IMDb
def get_imdb_reviews(movie_id):
    url = f"https://www.imdb.com/title/{movie_id}/reviews"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return f"HTTP Error: {response.status_code}"

    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = soup.find_all('div', class_='text show-more__control')

    if not reviews:
        return "No reviews found or scraping is blocked."

    reviews_list = [review.get_text() for review in reviews]
    return reviews_list

# Função para remover tags <br /> de um texto
def clean_review(text):
    return re.sub(r'<br\s*/?>', ' ', text)

# Função para plotar a distribuição dos sentimentos
def plot_sentiment_distribution(movie_title, sentiments):
    plt.figure(figsize=(6,4))
    plt.hist(sentiments, bins=3, edgecolor='black')
    plt.title(f"Sentiment Distribution for {movie_title}")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.xticks([0, 1], ['Negative', 'Positive'])
    st.pyplot(plt)

# Configurações do Streamlit
st.title("IMDb Movie Review Sentiment Analysis")

# Entrada do usuário para o ID do filme
movie_id = st.text_input("Enter IMDb movie ID:")

# Verifica se o ID foi fornecido
if movie_id:
    movie_title = st.text_input("Enter Movie Title:")
    
    # Carregar o dataset de treino
    df = pd.read_csv('IMDB Dataset.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Limpar as resenhas
    df['review'] = df['review'].apply(clean_review)

    # Separar features e labels para treino
    X_train = df['review']
    y_train = df['sentiment']

    # Criar um pipeline de vetorização e modelo
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Obter reviews do IMDb
    reviews = get_imdb_reviews(movie_id)

    if isinstance(reviews, list):
        sentiments = []
        for review in reviews:
            cleaned_review = clean_review(review)
            sentiment = model.predict([cleaned_review])[0]
            sentiments.append(sentiment)
        
        # Mostrar as reviews e sentimentos
        for i, review in enumerate(reviews, start=1):
            sentiment_str = 'positive' if sentiments[i-1] == 1 else 'negative'
            st.write(f"Review {i}:")
            st.write(review)
            st.write(f"Sentiment: {sentiment_str}")
            st.write("-" * 80)
        
        # Plotar a distribuição dos sentimentos
        plot_sentiment_distribution(movie_title, sentiments)
    else:
        st.write(reviews)