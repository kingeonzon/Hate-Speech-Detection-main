# iknowhate.py

import streamlit as st
import matplotlib.pyplot as plt
import pickle 
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from data import generate_wordcloud_from_csv
import requests
import time
import random

# Define a basic clean function
def clean(text):
    return text.strip().lower()

def load_tfidf():
    tfidf = pickle.load(open("tf_idf1.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model1.pkt", "rb"))
    return nb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

st.markdown("<h1 class='centered-title'>Toxicity Detection App</h1>", unsafe_allow_html=True)


st.markdown("""
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .centered-title {
        text-align: center;
        box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        padding: 10px;
    }
    .centered-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .centered-button button {
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("<h5 style='text-align: center; margin-top: 10px;'>Input any type of text in the text area to detect if it is toxic or not.</h5>",unsafe_allow_html=True)

st.markdown("""
    <style>
        .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)


def fetch_facts():
    api_url = 'https://api.api-ninjas.com/v1/facts'
    response = requests.get(api_url, headers={'X-Api-Key': 'W7rQ8SWVPhD/7Dy2K9Fv1Q==f3meAYiK48e0NOoB'})
    if response.status_code == 200:
        return response.json()
    else:
        return None


about_content = """
    <h1 style='text-align: center; margin-top: 30px;'>Welcome</h1>
    <h5 style='text-align: center; margin-top: 10px;'>The aim of this tool is to detect harmful passages from any text.</h4>
    <h5 style='text-align: center;'>Upload your text and click 'Detect' to get started.</h4>
    <h4 style='text-align: center; margin-top: 50px'>Remember to always be careful with your words because you'll never know it will affect someone.</h4>
    <h5 style='text-align: center; margin-top: 100px;'>Project by: Geonzon, Doriquez, and Cajegas</h4> 
"""
facts_content = """"
"""

# Sidebar
with st.sidebar:
    st.subheader("Navigation")
    option = st.radio("Go To",["About", "Facts"])

    if option == "About":
        st.markdown(about_content, unsafe_allow_html=True)
    elif option == "Facts":
        st.markdown("<h1 style='text-align: center; margin-top: 50px;'>Did You Know</h1>", unsafe_allow_html=True)
        fact_placeholder = st.empty()  # Placeholder for displaying the fact
        
        if st.button("Get New Fact"):
            facts = fetch_facts()  # Call the fetch_facts function
            if facts:
                random_fact = random.choice(facts)['fact']  # Select a random fact
                fact_placeholder.write(random_fact)
            else:
                fact_placeholder.error("Failed to fetch facts from the API. Please try again later.") 
        


text_input = st.text_area("Enter your text")  # Changed from st.text_input to st.text_area

if text_input:
    if st.button("Detect"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        if result == "Toxic":
            st.error("The result is Toxic.")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("hate_speech_image.gif", caption="Why did u have to make him cry :<", width=300)
        else:
            st.success("The result is Non-Toxic.")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("happy.gif", caption="yeeeeeeyyyyyy!!!!!", width=300)

st.markdown("---")

st.markdown("<h1 class='centered-title'>Twitter Data Word Cloud</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; margin-top: 10px; margin-bottom:10px;'>See the visualized frequency of words used in the dataset.</h5>",unsafe_allow_html=True)


st.markdown("<div class='centered-button'>", unsafe_allow_html=True)
if st.button("Generate Word Cloud", key="generate_button"):
    try:
        wordcloud = generate_wordcloud_from_csv('FinalBalancedDataset.csv')  # Call the function to generate the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)  # Display the word cloud using Streamlit's pyplot
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
st.markdown("</div>", unsafe_allow_html=True)
