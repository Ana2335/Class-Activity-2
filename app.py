import streamlit as st
import image
import regression
#import text4

st.title("Deep learning app 🧠")

# Menú principal
option = st.selectbox("Select a model to use:",
    ("Image Classifier", "Regression (Median house value)", "Text Classifier (Review sentiment)"))

# Mostrar la página correspondiente según la selección
if option == "Image Classifier":
    image.main()
elif option == "Regression (Median house value)":
    regression.main()
#elif option == "Text Classifier (Review sentiment)":
#    text4.main()
