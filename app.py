import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit app
st.title("Mental Health Companion Chatbot")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if chat history is empty and add initial message from model
if not st.session_state.messages:
    initial_message = {"role": "model", "content": "Hey there! How's your day been?"}
    st.session_state.messages.append(initial_message)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Talk to me..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Priming the model for compassionate responses:
        compassionate_prompt = f"""You are a compassionate and empathetic AI assistant.  A user has shared the following: '{prompt}'. Please respond in a way that is supportive, understanding, and validates their feelings.  Offer helpful suggestions if appropriate, but prioritize being a good listener and showing genuine care."""

        for response in model.generate_content(
            compassionate_prompt,  # Using the modified prompt
            stream=True,
        ):
            full_response += response.text
            # Limit response length
            if len(full_response) > 2000:
                full_response = full_response[:2000] + "..."
                break
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )