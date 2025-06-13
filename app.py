import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Mobile Price Predictor", layout="centered")
st.title("📱 Mobile Price Range Predictor")
st.markdown("Predict if a mobile phone is **Low**, **Medium**, **High**, or **Very High** priced based on its features.")

# Sidebar inputs
st.sidebar.header("📋 Enter Mobile Specs")

def yes_no(label):
    return st.sidebar.radio(label, ["No (0)", "Yes (1)"])

# Get input features
battery_power = st.sidebar.slider("🔋 Battery Power (mAh)", 500, 2000, 1000)
blue = yes_no("🔵 Bluetooth")
clock_speed = st.sidebar.slider("🕒 Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
dual_sim = yes_no("📶 Dual SIM")
fc = st.sidebar.slider("🤳 Front Camera (MP)", 0, 20, 5)
four_g = yes_no("📡 4G Support")
int_memory = st.sidebar.slider("💾 Internal Memory (GB)", 2, 128, 32)
m_deep = st.sidebar.slider("📏 Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.01)
mobile_wt = st.sidebar.slider("⚖️ Weight (gm)", 80, 250, 150)
n_cores = st.sidebar.slider("🧠 Processor Cores", 1, 8, 4)
pc = st.sidebar.slider("📸 Primary Camera (MP)", 0, 30, 13)
px_height = st.sidebar.slider("📐 Pixel Height", 0, 1960, 960)
px_width = st.sidebar.slider("📐 Pixel Width", 500, 2000, 1200)
ram = st.sidebar.slider("🧠 RAM (MB)", 256, 8192, 4096, step=128)
sc_h = st.sidebar.slider("📲 Screen Height (cm)", 5, 20, 10)
sc_w = st.sidebar.slider("📲 Screen Width (cm)", 2, 12, 5)
talk_time = st.sidebar.slider("🔋 Talk Time (hr)", 2, 20, 10)
three_g = yes_no("📶 3G Support")
touch_screen = yes_no("🖱️ Touch Screen")
wifi = yes_no("📡 WiFi Support")

# Convert yes/no to binary
def convert(value):
    return 1 if "Yes" in value else 0

# Create input array
input_data = np.array([[
    battery_power,
    convert(blue),
    clock_speed,
    convert(dual_sim),
    fc,
    convert(four_g),
    int_memory,
    m_deep,
    mobile_wt,
    n_cores,
    pc,
    px_height,
    px_width,
    ram,
    sc_h,
    sc_w,
    talk_time,
    convert(three_g),
    convert(touch_screen),
    convert(wifi)
]])

# Predict
if st.button("🔍 Predict Price Range"):
    prediction = model.predict(input_data)[0]
    labels = ["Low", "Medium", "High", "Very High"]
    st.success(f"📊 Predicted Price Range: **{labels[prediction]}**")
