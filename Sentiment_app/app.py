import streamlit as st
import pickle

# Load pre-trained model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("📊 Real-Time Sentiment Analysis")
st.write("Enter a review below and get sentiment prediction (Positive/Negative).")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        
        if len(user_input.split()) <= 1:
            st.warning("Text too short → likely Neutral")
        else:
            review_vect = vectorizer.transform([user_input])
            prediction = model.predict(review_vect)[0]
            st.success(f"Predicted Sentiment: **{prediction}**")
    
    else:
        st.error("Please enter some text to analyze.")