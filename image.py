import streamlit as st
import pickle
import numpy as np
from PIL import Image

def main():

    etiquetas = {0: "Plane ✈️", 1: "Automobile 🚗", 2: "Bird 🐦‍⬛", 3: "Cat 🐱", 4: "Deer 🦌",
                5: "Dog 🐶", 6: "Frog 🐸", 7: "Horse 🐴", 8: "Ship ⛵️", 9: "Truck 🚚"}

    # Subir modelo
    picklefile = open("image2.pkl", "rb")
    model_image = pickle.load(picklefile)

    st.header("Image Classifier 📷")
    st.write("Upload an image to predict its category")
    img = st.file_uploader("Enter an image", type=["jpg", "jpeg", "png"])

    if img:
        # Cargar la imagen
        image = Image.open(img).convert("RGB")
        st.image(image, caption="Image uploaded", use_column_width=True)

        # Procesamiento de la imagen
        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 32 * 32 * 3)

        # Predicción
        if st.button("Predict"):
            prediction = model_image.predict(image_array)
            predicted_class = int(np.argmax(prediction, axis=1))
            st.success(f"Predicted category: {etiquetas[predicted_class]}")
            #st.write(f"Predicted category: {predicted_class}: {etiquetas[predicted_class]}")
            #st.write(f"Categoría predicha: **{categorías[predicted_class]}**")
            #st.write(f"La categoría {predicted_class} es: {etiquetas[predicted_class]}")
