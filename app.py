import os
import streamlit as st
from dotenv import load_dotenv
import cv2
from emotion_detection import detect_emotion
from chatbot import initialize_chatbot, generate_response, extract_query_from_response, search_youtube
from utils.voice_input import get_voice_input
from utils.text_to_speech import speak

# Load environment variables
load_dotenv()

# Initialize the chatbot
groq_chat, app, config = initialize_chatbot()

# Streamlit app
st.title("Mental Health Companion Chatbot")

# Sidebar for YouTube recommendations
st.sidebar.title("YouTube Recommendations")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    initial_message_content = "Hey there! How's your day been? \U0001F60A"
    initial_message = {"role": "assistant", "content": initial_message_content}
    st.session_state.messages.append(initial_message)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Detect emotion from the frame
        frame, emotion_label = detect_emotion(frame)
        st.image(frame, channels="BGR")
    cap.release()
else:
    st.error("Failed to open webcam.")

# Unified input handling
def handle_input(user_input):
    if user_input:
        # Display the user's input in the chat interface
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add the user's input to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate a response
        full_response = generate_response(
            user_input, emotion_label, groq_chat, app, config, st.session_state.messages
        )

        # Extract the chatbot's message (before the delimiter)
        if "|||" in full_response:
            chatbot_message = full_response.split("|||", 1)[0].strip()
        else:
            chatbot_message = full_response

        # Display the assistant's response (without the query)
        with st.chat_message("assistant"):
            st.markdown(chatbot_message)

        # Add the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": chatbot_message})

        # Extract the YouTube query from the chatbot's response
        youtube_query = extract_query_from_response(full_response)

        # Fetch YouTube recommendations only if a valid query is found
        if youtube_query:
            youtube_results = search_youtube(youtube_query)

            # Display YouTube recommendations in the sidebar
            if youtube_results:
                st.sidebar.markdown("### Recommended Videos")
                for video in youtube_results:
                    st.sidebar.markdown(f"- [{video['title']}]({video['url']})")
            else:
                st.sidebar.markdown("No videos found for the given query.")
        else:
            st.sidebar.markdown("No YouTube query found in the chatbot's response.")

        # Convert the assistant's response to speech
        speak(chatbot_message)  # Speak the assistant's response

# Voice input button
if st.button("🎤 Use Voice Input"):
    user_input = get_voice_input()
    handle_input(user_input)

# Text input (existing chat input)
if prompt := st.chat_input("Talk to me..."):
    handle_input(prompt)