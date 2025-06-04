import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from datasets import load_dataset

def main():

    imdb = load_dataset("imdb")

    st.header("Movie Review Sentiment Classifier 🎬💕")

    # Cargar modelo 
    @st.cache_resource
    def load_model():
        with open("text_4.pkl", "rb") as f:
            return pickle.load(f)

    model_text = load_model()

    # Tokenizer basado en IMDB
    vocab_size = 10000
    max_len = 500
    #word_index = imdb.get_word_index()
    #tokenizer = Tokenizer(num_words=vocab_size)
    #tokenizer.word_index = word_index
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(imdb['train']['text'])

    # Entrada del usuario
    text = st.text_area("Write a movie review:", "This movie was terrible")

    if st.button("Predict"):
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_len)

        # Predicción (vector de probabilidad de 2 clases)
        prediction = model_text.predict(padded)[0]
        predicted_class = np.argmax(prediction)
        #prob = prediction[predicted_class]

        if predicted_class == 1:
            st.success(f"Positive review 😄")
        else:
            st.error(f"Negative review 😞")
        #label_map = {0: "Negativa 😞", 1: "Positiva 😄"}
        #st.markdown(f"### Resultado: {label_map[predicted_class]}")
        #st.write(f"**Confianza:** {prob:.2f}")
