# api.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from datasets import load_dataset

app = Flask(__name__)

# Cargar modelo y tokenizer
with open("text_4.pkl", "rb") as f:
    model_text = pickle.load(f)

# Preparamos el tokenizer con el dataset IMDB (igual que en tu app)
imdb = load_dataset("imdb")
vocab_size = 10000
max_len = 500
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(imdb['train']['text'])

@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data['text']
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)

    prediction = model_text.predict(padded)[0]
    predicted_class = int(np.argmax(prediction))

    return jsonify({
        "text": text,
        "predicted_class": predicted_class,
        "sentiment": "positive" if predicted_class == 1 else "negative"
    })

if __name__ == '__main__':
    app.run(debug=True)
