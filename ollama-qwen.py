import os
import cv2
import streamlit as st
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import ollama  # Import Ollama
import smtplib  # For email escalation
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import speech_recognition as sr  # For voice input
import pyttsx3  # For voice output

# Load environment variables
load_dotenv()

# Load the trained model
model_best = load_model('model.h5')  # Set your machine model file path here

# Classes for 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Email configuration (from .env or hardcoded)
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ESCALATION_EMAIL = os.getenv("ESCALATION_EMAIL")  # Email to notify in case of escalation

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
    emotion_label = 'Neutral'

# Function to send escalation email
def send_escalation_email(emotion, user_input):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = ESCALATION_EMAIL
        msg['Subject'] = f"Emotion Escalation: User is feeling {emotion}"
        body = f"""
        The user is currently feeling {emotion}.
        Their recent input was: "{user_input}".
        Please reach out to them as soon as possible.
        """
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, ESCALATION_EMAIL, msg.as_string())
        st.success(f"Escalation email sent to {ESCALATION_EMAIL}.")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Function to get voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            st.success("Voice input received.")
            return user_input
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return None

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to decide if escalation is needed
def should_escalate(emotion, user_input):
    # Define conditions for escalation
    negative_emotions = ['Angry', 'Disgusted', 'Fear', 'Sad']
    if emotion in negative_emotions and any(keyword in user_input.lower() for keyword in ["help", "sad", "angry", "scared", "depressed"]):
        return True
    return False

# Voice input button
st.write("### Voice Input")
if st.button("🎤 Use Voice Input"):
    user_input = get_voice_input()
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check if escalation is needed
        if should_escalate(emotion_label, user_input):
            send_escalation_email(emotion_label, user_input)
            st.warning("Escalation triggered due to negative emotion and concerning input.")

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            compassionate_prompt = f"""You are a compassionate and empathetic AI assistant. A user has shared the following: '{user_input}'.The user is feeling '{emotion_label}'. Please respond in a way that is supportive, understanding, and validates their feelings. Use emotes to convey emotions. Offer helpful suggestions if appropriate, but prioritize being a good listener and showing genuine care. 😊"""

            # Use Ollama's API to generate a response
            response = ollama.generate(model="qwen2.5:0.5b", prompt=compassionate_prompt, stream=True)
            
            for chunk in response:
                full_response += chunk['response']
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Convert the response to speech
            speak(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Text input (existing chat input)
if prompt := st.chat_input("Talk to me..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if escalation is needed
    if should_escalate(emotion_label, prompt):
        send_escalation_email(emotion_label, prompt)
        st.warning("Escalation triggered due to negative emotion and concerning input.")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        compassionate_prompt = f"""You are a compassionate and empathetic AI assistant. A user has shared the following: '{prompt}'.The user is feeling '{emotion_label}'. Please respond in a way that is supportive, understanding, and validates their feelings. Use emotes to convey emotions. Offer helpful suggestions if appropriate, but prioritize being a good listener and showing genuine care. 😊"""

        # Use Ollama's API to generate a response
        response = ollama.generate(model="qwen2.5:0.5b", prompt=compassionate_prompt, stream=True)
        
        for chunk in response:
            full_response += chunk['response']
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        # Convert the response to speech
        speak(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})