import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load model and vectorizer
model = joblib.load("linear_svc_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")  # You must save this too during training
label_encoder = joblib.load("label_encoder.pkl")  # Save the LabelEncoder as well

# Streamlit UI
st.title("🧠 Emotion Classifier")
st.write("Enter a sentence and find out the predicted emotion.")

user_input = st.text_area("Input Text", "He was afraid of the dark.")

if st.button("Predict Emotion"):
    cleaned_input = preprocess(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)
    emotion = label_encoder.inverse_transform(prediction)[0]
    st.success(f"**Predicted Emotion:** {emotion}")
