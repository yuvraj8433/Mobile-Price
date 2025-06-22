import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(page_title="📱 Mobile Price Predictor", layout="centered")

st.title("📱 Mobile Price Range Predictor")
st.markdown("Enter mobile features to predict if it is **Low**, **Medium**, **High**, or **Very High** priced.")

st.divider()

# Layout input using columns
col1, col2 = st.columns(2)

with col1:
    battery_power = st.slider("🔋 Battery Power (mAh)", 500, 2000, 1000)
    ram = st.slider("🧠 RAM (MB)", 256, 8192, 4096, step=128)
    px_height = st.slider("📏 Pixel Height", 0, 1960, 960)
    four_g = st.selectbox("📡 4G Support", ["No (0)", "Yes (1)"])

with col2:
    px_width = st.slider("📏 Pixel Width", 500, 2000, 1200)
    touch_screen = st.selectbox("🖱️ Touch Screen", ["No (0)", "Yes (1)"])
    wifi = st.selectbox("📶 WiFi Support", ["No (0)", "Yes (1)"])

# Convert function
def convert(value):
    return 1 if "Yes" in value else 0

# Input array
input_data = np.array([[battery_power, ram, px_height, px_width,
                        convert(four_g), convert(touch_screen), convert(wifi)]])

# Predict and show result
if st.button("🔍 Predict Price Range"):
    prediction = model.predict(input_data)[0]
    labels = ["Low", "Medium", "High", "Very High"]
    st.success(f"📊 Predicted Price Range: **{labels[prediction]}**")
