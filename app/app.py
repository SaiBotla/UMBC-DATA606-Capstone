#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import warnings
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Exploratory Data Analysis
ted_main = pd.read_csv('ted_main.csv')
trans = pd.read_csv('transcripts.csv')

# Number of words in each talk
trans['number_of_words'] = trans['transcript'].str.split().str.len()
fig1 = px.histogram(trans, x="number_of_words", hover_data=trans.columns)

# Count of Main Speakers
main_speaker_count = ted_main['main_speaker'].value_counts().head(50)
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.countplot(data=ted_main, y="main_speaker", order=main_speaker_count.index, ax=ax2)

# Speakers Occupation
speaker_occupation_count = ted_main['speaker_occupation'].value_counts().head(25)
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.countplot(data=ted_main, y="speaker_occupation", order=speaker_occupation_count.index, ax=ax3)

# Clustering
from sklearn.cluster import KMeans

# Load the data
ted_main_df = pd.read_csv('ted_main.csv')
transcripts_df = pd.read_csv('transcripts.csv')

# Merge the data frames on URL
# ted_data_df = pd.merge(ted_main_df, transcripts_df, on='url')
#
# # Data Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# ted_data_df['processed_transcript'] = ted_data_df['transcript'].apply(preprocess_text)
#
# # Apply TF-IDF vectorization
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(ted_data_df['processed_transcript'])
#
# # Apply word embeddings
# word_embeddings = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#
# # Matrix Factorization
ted_main = pd.read_csv('ted_main.csv')
transcripts = pd.read_csv('transcripts.csv')
ted_data = pd.merge(left=ted_main, right=transcripts, how='left', left_on='url', right_on='url')
#
# # Preprocessing
# ted_data.dropna(inplace=True)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = pos_tag(stemmed_tokens)
    lemmatized_tokens = []
    for token, tag in tagged_tokens:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_tokens.append(lemmatizer.lemmatize(token, pos))
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# ted_data['processed_text'] = ted_data['transcript'].apply(preprocess_text)
#
# # Training
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(ted_data['processed_text'])
# similarities = cosine_similarity(tfidf_matrix)

import numpy as np
# Save the array to a file in binary format
# np.save('similarities.npy', similarities)

# Load the array from the saved file
similarities = np.load('similarities.npy')
st.write(similarities)

# Function to get recommended talks based on title
def recommend(title):
    index = ted_data[ted_data['title'] == title].index[0]
    cosine_similarities = similarities[index]
    top_indices = cosine_similarities.argsort()[-6:-1][::-1]
    recommended_talks = ted_data.iloc[top_indices]['title']
    return recommended_talks

# Streamlit App
def main():
    st.title('TED Talk Analyzer')

    # Display number of words histogram
    st.subheader('Number of Words in Talks')
    st.plotly_chart(fig1)

    # Display count of main speakers
    st.subheader('Count of Main Speakers')
    st.pyplot(fig2)

    # Display count of speaker occupations
    st.subheader('Speakers Occupation')
    st.pyplot(fig3)

    # Input for title
    title_input = st.text_input('Enter the title of the talk:', 'How to speak so that people want to listen')

    # Get recommendations
    recommendations = recommend(title_input)

    # Display recommendations
    st.subheader('Recommended Talks')
    for i, talk in enumerate(recommendations, start=1):
        st.write(f"{i}. {talk}")

if _name_ == "_main_":
    main()
