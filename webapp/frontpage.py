import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import requests
from twilio.rest import Client
import os 

# Twilio API Credentials
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
EMERGENCY_CONTACT =os.getenv("EMERGENCY_CONTACT")

# Load trained model
model = load_model("models/trained.h5")

# Streamlit UI Styling
st.set_page_config(page_title="Accident Detection System", page_icon="🚨", layout="wide")
st.markdown("""
    <style>
        .main-title {
            font-size: 100px;
            font-weight: bold;
            text-align: center;
            color: #FF4B4B;
        }

        .sub-title {
            font-size: 30px;
            text-align: center;
            color: #4A90E2;
        }

        .sidebar .sidebar-content {
            background-color: #FFD1DC !important;
            padding: 20px;
            border-radius: 10px;
        }

        p {
            font-size: 60px;
        }

        body {
            background-color: #FFF3E0;
        }

        .main {
            background-color: #FFF3E0 !important;
        }

        .stApp {
            background-color: #FFF3E0;
        }

        /* Make prediction output text black */
        .stAlert > div {
            color: black !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)



# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Upload Image",  "CCTV Feed"])
st.markdown('<p style="font-size:50px; font-weight:bold; text-align:center; color:#FF4B4B;">Accident Detection System 🚨</p>', unsafe_allow_html=True)

st.markdown('<p style="font-size:30px; text-align:center; color:#4A90E2; font-weight:600;">Using Deep learning for real-time accident detection</p>', unsafe_allow_html=True)


# Upload Image Mode
if app_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(255, 255))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 255.0  

        y_pred = model.predict(img_array)
        
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if y_pred < 0.5:
            st.error("🚨 Accident Detected! Sending Emergency Alert...")
        else:
            st.success("✅ No Accident Detected")



# CCTV Feed Mode
elif app_mode == "CCTV Feed":
    st.title("📡 Live Public CCTV Feed")
    cctv_url = "http://207.251.86.238/cctv1001.jpg"
    st.write("Fetching live feed...")
    frame_window = st.image([])
    
    while True:
        response = requests.get(cctv_url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
            processed_frame = cv2.resize(frame, (255, 255)) / 255.0
            processed_frame = np.expand_dims(processed_frame, axis=0)
            prediction = model.predict(processed_frame)[0][0]
            
            label = "✅ No Accident" if prediction > 0.5 else "❌ Accident Detected"
            color = (0, 255, 0) if prediction > 0.5 else (255, 0, 0)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_window.image(frame)
    
st.sidebar.write("Developed with using Streamlit")