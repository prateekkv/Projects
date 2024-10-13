# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:35:24 2024

@author: dekrk
"""

import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import pickle

# Load the saved model and TF-IDF vectorizer
loaded_model = pickle.load(open('lrmodel.pkl', 'rb'))
tfidf = pickle.load(open('TFIDF2.pkl', 'rb'))

#------------------------------------------------------------------------------

# Text Pre-processing
wordnet = WordNetLemmatizer()
stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

not_stopwords = ("aren", "aren't", "couldn", "couldn't", "didn", "didn't",
                 "doesn", "doesn't", "don", "don't", "hadn", "hadn't", "hasn",
                 "hasn't", "haven", "haven't", "isn", "isn't", "mustn",
                 "mustn't", "no", "not", "only", "shouldn", "shouldn't",
                 "should've", "wasn", "wasn't", "weren", "weren't", "will",
                 "wouldn", "wouldn't", "won't", "very")
stop_words_ = [words for words in stop_words if words not in not_stopwords]
stop_words_.append("I")
stop_words_.append("the")
stop_words_.append("s")

# Function to preprocess user input
def preprocess_text(text):
    # Removal of puntuations
    review = re.sub('[^a-zA-Z]', ' ', text)

    # Converting Text to Lower case
    review = review.lower()

    # Spliting each words
    review = review.split()

    # Applying Lemmitization for the words 
    review = nlp(' '.join(review))
    review = [token.lemma_ for token in review]

    # Removal of stop words
    review = [word for word in review if word not in stop_words_]

    # Joining the words in sentences
    review = ' '.join(review)
    return review

#-----------------------------------------------------------------------------
# Streamlit app
st.title("Model Deployment:Hotel Review - Sentiment Analysis ")
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.markdown('This is a very simple webapp for sentiment analysis of Hotel Review with Logistic Regression ')


user_input = st.text_area("Enter your review:")
if st.button("Analyze"):
    if user_input:
        # Preprocess the user input
        processed_input = preprocess_text(user_input)

        # Transform the preprocessed input using TF-IDF
        input_tfidf = tfidf.transform([processed_input])

        # Make prediction
        prediction = loaded_model.predict(input_tfidf)[0]

        # Print the prediction
        if prediction == 1:
            st.write("Positive Feedback")
        else:
            st.write("Negative Feedback")
    else:
        st.write("Please enter a review.")
#------------------------------------------------------------------------------