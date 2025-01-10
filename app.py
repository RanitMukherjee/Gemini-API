import os
import cv2
import streamlit as st
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
import speech_recognition as sr  # For voice input
import pyttsx3  # For voice output
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid

# Load environment variables
load_dotenv()

# Load the trained model
model_best = load_model('model.h5')  # Set your machine model file path here

# Classes for 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_chat = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key)

# System prompt for the chatbot
system_prompt = """You are a compassionate and empathetic AI assistant. The user is feeling '{emotion_label}'. Please respond in a way that is supportive, understanding, and validates their feelings. Use emotes to convey emotions. Offer helpful suggestions if appropriate, but prioritize being a good listener and showing genuine care. 😊"""

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

emotion_label = 'Neutral'
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

# Function to convert text to speech (interrupt-tolerant)
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        # If the TTS engine is already running, stop it and restart
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    finally:
        engine.stop()  # Ensure the engine is stopped after speaking

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    selected_messages = state["messages"]
    response = groq_chat.invoke(selected_messages)
    return {"messages": [response]}

# Define the two nodes we will cycle between
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Adding memory is straight forward in langgraph!
memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory
)

# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Unified function to handle user input (both voice and text)
def handle_user_input(user_input):
    if user_input:
        # Add user input to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Construct the chat prompt template
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt.format(emotion_label=emotion_label)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            # Create a conversation chain
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=False,
            )

            # Prepare the chat history
            chat_history = [HumanMessage(content=msg["content"]) for msg in st.session_state.messages if msg["role"] == "user"]

            # Generate a response using the conversation chain
            response = conversation.predict(human_input=user_input, chat_history=chat_history)
            full_response = response

            message_placeholder.markdown(full_response)

            # Convert the response to speech
            speak(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Voice input button
if st.button("🎤 Use Voice Input"):
    user_input = get_voice_input()
    handle_user_input(user_input)

# Text input (existing chat input)
if prompt := st.chat_input("Talk to me..."):
    handle_user_input(prompt)