# -*- coding: utf-8 -*-
"""Cópia de Prova Final Roriz

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MDlcGdTOvFrDY-U937hpXO3YEtweJKbI
"""

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
from wordcloud import WordCloud

df = pd.read_csv('/content/drive/MyDrive/IMDB Dataset.csv')

import re
def clean_review(text):
    return re.sub(r'<br\s*/?>', '', text)

# Aplicar a função a todas as resenhas na coluna 'review'
df['review'] = df['review'].apply(clean_review)

# Exibir as primeiras linhas do DataFrame para verificar
print(df.head())

# Visualizar informações do dataframe
print(df.info())

# Visualizar a distribuição das classes
print(df['sentiment'].value_counts())

print(df.describe())
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df)
plt.title('Distribuição das Classes')
plt.show()

df['review_length'] = df['review'].apply(len)

# Estatísticas descritivas dos comprimentos das resenhas
print(df['review_length'].describe())

# Plotar a distribuição do comprimento das resenhas
plt.figure(figsize=(12,6))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title('Distribuição do Comprimento das Resenhas')
plt.xlabel('Comprimento da Resenha')
plt.ylabel('Frequência')
plt.show()

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Resenhas positivas e negativas
positive_reviews = ' '.join(df[df['sentiment']=='positive']['review'])
negative_reviews = ' '.join(df[df['sentiment']=='negative']['review'])

# Wordcloud para resenhas positivas
plot_wordcloud(positive_reviews, 'Wordcloud - Resenhas Positivas')

# Wordcloud para resenhas negativas
plot_wordcloud(negative_reviews, 'Wordcloud - Resenhas Negativas')

dfDois = pd.read_csv('/content/drive/MyDrive/imdb-reviews-pt-br.csv')

dfDois.head()

dfDois['sentiment'] = dfDois['sentiment'].replace({'neg': 'negative', 'pos': 'positive'})
dfDois = dfDois.drop(columns=['text_pt'])
dfDois = dfDois.drop(columns=['id'])
dfDois = dfDois.rename(columns={'text_en': 'review'})

# Visualizar informações do dataframe
print(dfDois.info())

# Visualizar a distribuição das classes
print(dfDois['sentiment'].value_counts())

print(dfDois.describe())
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=dfDois)
plt.title('Distribuição das Classes')
plt.show()

dfDois['review_length'] = dfDois['review'].apply(len)

# Estatísticas descritivas dos comprimentos das resenhas
print(dfDois['review_length'].describe())

# Plotar a distribuição do comprimento das resenhas
plt.figure(figsize=(12,6))
sns.histplot(dfDois['review_length'], bins=50, kde=True)
plt.title('Distribuição do Comprimento das Resenhas')
plt.xlabel('Comprimento da Resenha')
plt.ylabel('Frequência')
plt.show()

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Resenhas positivas e negativas
positive_reviews = ' '.join(dfDois[dfDois['sentiment']=='positive']['review'])
negative_reviews = ' '.join(dfDois[dfDois['sentiment']=='negative']['review'])

# Wordcloud para resenhas positivas
plot_wordcloud(positive_reviews, 'Wordcloud - Resenhas Positivas')

# Wordcloud para resenhas negativas
plot_wordcloud(negative_reviews, 'Wordcloud - Resenhas Negativas')

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


def plot_sentiment_distribution(movie_title, sentiments):
    plt.figure(figsize=(6,4))
    plt.hist(sentiments, bins=3, edgecolor='black')
    plt.title(f"Sentiment Distribution for {movie_title}")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.show()


    # Carregar o dataset de treino


    # Pré-processamento dos dados
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
dfDois['sentiment'] = dfDois['sentiment'].map({'positive': 1, 'negative': 0})

X_train = df['review']
y_train = df['sentiment']

    # Separar features e labels para teste
X_test = dfDois['review']
y_test = dfDois['sentiment']

    # Criar um pipeline de vetorização e modelo
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Treinar o modelo
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {accuracy:.2%}')

movie_ids = {
        "Diário de uma Paixão": "tt0332280",
        "Forrest Gump": "tt0109830",
        "Batman: O Cavaleiro das Trevas": "tt0468569"
}

def main():

  for movie_title, movie_id in movie_ids.items():
        print(f"Reviews for {movie_title}:")
        reviews = get_imdb_reviews(movie_id)

        if isinstance(reviews, list):
            sentiments = []
            for review in reviews:
                sentiment = model.predict([review])[0]
                sentiments.append(sentiment)

            # Plot the sentiment distribution
            plot_sentiment_distribution(movie_title, sentiments)
        else:
            print(reviews)

  for movie_title, movie_id in movie_ids.items():
        print(f"Reviews for {movie_title}:")
        reviews = get_imdb_reviews(movie_id)

        if isinstance(reviews, list):
            for i, review in enumerate(reviews, start=1):
                sentiment = model.predict([review])[0]
                sentiment_str = 'positive' if sentiment == 1 else 'negative'
                print(f"Review {i}:")
                print(review)
                print(f"Sentiment: {sentiment_str}")
                print("-" * 80)
        else:
            print(reviews)
        print("=" * 80)

if __name__ == "__main__":
    main()