import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the trained pipeline
pipeline = joblib.load('model.pkl')

# Function to predict class
def predict_class(text):
    prediction = pipeline.predict([text])
    return prediction[0]

# Streamlit UI

st.markdown("""
    <div style='background-color:#f9f9f9;padding:10px;border-radius:10px'>
    <h2 style='text-align:center;color:#dc143c;'>Welcome to FarmAds Classifier</h2>
    <p style='text-align:justify;color:#5f6368;'>FarmAds Classifier is a web app for classifying classified ads on a website catering to a specific farming community. The goal is to automatically distinguish between relevant and irrelevant ads, addressing issues like fraud, spam, and lack of relevance..</p>
    </div>
    """, unsafe_allow_html=True)

# st.title('FarmAds Classifier')
# st.write('## Description')
# st.write("The problem involves developing a predictive model for classifying classified ads on a website catering to a specific farming community. The goal is to automatically distinguish between relevant (1) and irrelevant (-1) ads, addressing issues like fraud, spam, and lack of relevance.")

st.write('## Classify Ad')
ad_text = st.text_area('Enter the ad text here:')
if st.button('Submit'):
    if ad_text:
        prediction = predict_class(ad_text)
        if prediction == 1:
            st.success('The ad is relevant!')
        else:
            st.error('The ad is irrelevant!')
    else:
        st.warning('Please enter some text to classify.')
