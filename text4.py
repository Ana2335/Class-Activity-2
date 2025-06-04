import streamlit as st
import pickle
import numpy as np

def main():
  
    st.header("Movie Review Sentiment Classifier 🎬💕")

    # Cargar modelo y vectorizador (TF-IDF + clasificador)
    @st.cache_resource
    def load_model():
        with open("text_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("text_classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        return vectorizer, classifier

    vectorizer, classifier = load_model()

    # Entrada del usuario
    text = st.text_area("Write a movie review:", "This movie was terrible")

    if st.button("Predict"):
        # Preprocesamiento y predicción
        X_input = vectorizer.transform([text])
        prediction = classifier.predict(X_input)[0]

        # Mostrar resultado
        if prediction == 1:
            st.success("Positive review 😄")
        else:
            st.error("Negative review 😞")
