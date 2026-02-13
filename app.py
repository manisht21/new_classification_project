import streamlit as st
import joblib

st.set_page_config(page_title="News Classifier", page_icon="📰")

st.title("📰 News Article Classification")
st.write("Enter a news article and get the predicted category.")

# Load trained pipeline (TF-IDF + LogisticRegression)
model = joblib.load("models/news_classifier.pkl")

# AG News label mapping
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

user_input = st.text_area("Enter news article text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        predicted_label = label_map.get(int(prediction), "Unknown")
        st.success(f"Predicted Category: {predicted_label}")
