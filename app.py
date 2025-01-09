import os
import cv2
import streamlit as st
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import ollama  # Import Ollama

# Load environment variables
load_dotenv()

# Load the trained model
model_best = load_model('model.h5')  # set your machine model file path here

# Classes 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit app
st.title("Mental Health Companion Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    initial_message_content = "Hey there! How's your day been? \U0001F60A"
    initial_message = {"role": "assistant", "content": initial_message_content}
    st.session_state.messages.append(initial_message)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y + h, x:x + w]

            # Resize the face image to the required input size for the model
            face_image = cv2.resize(face_roi, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = np.vstack([face_image])

            # Predict emotion using the loaded model
            predictions = model_best.predict(face_image)
            emotion_label = class_names[np.argmax(predictions)]

            # Display the emotion label on the frame
            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame in Streamlit
        st.image(frame, channels="BGR")

    # Release the webcam
    cap.release()
else:
    st.error("Failed to open webcam.")

# User input and AI response
if prompt := st.chat_input("Talk to me..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        compassionate_prompt = f"""You are a compassionate and empathetic AI assistant. A user has shared the following: '{prompt}'. Please respond in a way that is supportive, understanding, and validates their feelings. Use emotes to convey emotions. Offer helpful suggestions if appropriate, but prioritize being a good listener and showing genuine care. 😊"""

        # Use Ollama's API to generate a response
        response = ollama.generate(model="qwen2.5:0.5b", prompt=compassionate_prompt)
        full_response = response['response']

        if len(full_response) > 2000:
            full_response = full_response[:2000] + "..."

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})