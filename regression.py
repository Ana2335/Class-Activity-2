import streamlit as st
import pickle
import numpy as np

def main():

    # Cargar el modelo
    with open("regression.pkl", "rb") as model_file:
        model_regression = pickle.load(model_file)

    st.header("Predict the median house value for California districts 🏠")
    st.write("Enter the characteristics to estimate the average value of a home (in dollars)")

    # Características
    MedInc = st.number_input("Average household income")
    HouseAge = st.number_input("Average house age", step=1, format="%d")
    AveRooms = st.number_input("Average number of rooms per house", step=1, format="%d")
    AveBedrms = st.number_input("Average number of bedrooms per house", step=1, format="%d")
    Population = st.number_input("Block group population", step=1, format="%d")
    AveOccup = st.number_input("Average numer of household members", step=1, format="%d")
    Latitude = st.number_input("Latitud")
    Longitude = st.number_input("Longitud")

    # Predicción
    if st.button("Predict"):
        entrada = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])
        prediccion = model_regression.predict(entrada)
        valor = float(prediccion[0])
        st.success(f"Estimated value of the home: **${valor * 1000:,.2f}**")
        #st.write(f"Estimated value of the home: **${prediccion[0]*1000:,.2f}**")
