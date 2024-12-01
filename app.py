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

# Disclaimer (Crucial!)
st.warning(
    "**Disclaimer:** This chatbot is for informational and supportive purposes only. "
    "It is not a substitute for professional mental health advice. "
    "If you are experiencing a mental health crisis, please contact a qualified mental health professional or call a crisis hotline."
)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        for response in model.generate_content(
            prompt,
            stream=True,
            temperature=0.5,  # Moderate temperature for empathetic responses
        ):
            full_response += response.text
            # Limit response length to prevent excessively long responses.
            if len(full_response) > 2000:  # Adjust as needed
                full_response = full_response[:2000] + "..."
                break  # Stop generating further text
            message_placeholder.markdown(full_response + "▌")  # Typing indicator
        message_placeholder.markdown(full_response) # Finalize the response after streaming.
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )