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

model = load_model("C:\\Users\\Lenovo\\Desktop\\mlops_pro\\models\\trained.h5")


# Streamlit UI
st.title("Accident detection system")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])



if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(255, 255))  # Change (224, 224) based on model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required

    # Make prediction
    y_pred = model.predict(img_array)
    
    # Display results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if (y_pred<0.5):
        st.write("accident detected")
        def send_sms(message):
            client=Client(ACCOUNT_SID,AUTH_TOKEN)
            message=client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=EMERGENCY_CONTACT
            )
            return message.sid
        message_alert='Accident detected provide emergency services and ambulance facilities'
        message_id = send_sms(message_alert)
        st.success(f"Emergency SMS Sent! Message ID: {message_id}")
    else:
        st.write("everything is okay no accident detected")

def preprocess_frame(frame):
    frame = cv2.resize(frame, (255, 255))  # Resize to model input size
    frame = frame / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame
st.title('for live camera')
run=st.checkbox('start live')
run2=st.checkbox('cctv')
if run:
    cap=cv2.VideoCapture(0)
    stframe=st.empty()

    while True:
        ret,frame=cap.read()
        if not ret:
            print('falied to capture')
            break

        processed_frame=preprocess_frame(frame)
        prediction = model.predict(processed_frame)[0][0]  # Get probability

        # Display prediction result
        if prediction > 0.5:
            label = "No Accident ✅"
            color = (0, 255, 0)
            
        else:
            label = "Accident Detected ❌"
            color = (0, 0, 255)
            def send_sms(message):
                client=Client(ACCOUNT_SID,AUTH_TOKEN)
                message=client.messages.create(
                    body=message,
                    from_=TWILIO_PHONE_NUMBER,
                    to=EMERGENCY_CONTACT
                )
                return message.sid
            message_alert='Accident detected provide emergency services and ambulance facilities'
            message_id = send_sms(message_alert)
            st.success(f"Emergency SMS Sent! Message ID: {message_id}")

        # Show label on video
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show video in Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()
    
else:
      # Release camera if checkbox is unchecked
    cv2.destroyAllWindows()  # Close OpenCV windows
    st.warning("Live Camera Stopped! ✅")

if run2:
    st.title("Live Public CCTV Feed")

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
            processed_frame=preprocess_frame(frame)
            prediction = model.predict(processed_frame)[0][0]  # Get probability

        # Display prediction result
            if prediction > 0.5:
                label = "No Accident ✅"
                color = (0, 255, 0)
            else:
                label = "Accident Detected ❌"
                color = (0, 0, 255)
                def send_sms(message):
                    client=Client(ACCOUNT_SID,AUTH_TOKEN)
                    message=client.messages.create(
                        body=message,
                        from_=TWILIO_PHONE_NUMBER,
                        to=EMERGENCY_CONTACT
                    )
                    return message.sid
                message_alert='Accident detected provide emergency services and ambulance facilities'
                message_id = send_sms(message_alert)
                st.success(f"Emergency SMS Sent! Message ID: {message_id}")

        # Show label on video
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
