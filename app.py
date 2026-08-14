import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

@st.cache_resource
def load_model():
    return joblib.load("crop_model.pkl")

model = load_model()

st.title("🌾 Crop Recommendation System")
st.write("Soil aur weather values daalo, best crop recommend hoga.")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

if st.button("Recommend Crop", use_container_width=True):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(data)[0]
    
    st.success(f"**Recommended Crop:** {prediction}")
    st.balloons()
