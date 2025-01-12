# Mental Health Care Companion

## Overview

Mental Health Care Companion is an innovative application designed to support mental well-being through interactive text-based conversations and real-time emotional recognition using webcam/phone camera inputs. This project aims to provide users with empathetic responses, coping strategies, and emotional support based on their current emotional state.

## Features

- **Text-Based Interaction**: Engage in meaningful conversations with the companion to discuss feelings, thoughts, and receive supportive feedback.
- **Emotion Recognition**: Use webcam or phone camera to analyze facial expressions and detect emotions in real-time.
- **Personalized Support**: Receive tailored advice, resources, and coping strategies based on detected emotions.
- **Secure and Private**: All interactions and data are securely managed to ensure user privacy.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/RanitMukherjee/EmotiCare.git
    cd EmotiCare
    ```

2. **Create and Activate a Virtual Environment**:
    ```sh
    python -m venv myenv
    source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

### Environment Variables

Create a `.env` file in the root directory of the project and add the following environment variables:

```plaintext
API_KEY=your_api_key_here

### File Architecture

mental_health_chatbot/
│
├── app.py                  # Main Streamlit app
├── emotion_detection.py    # Emotion detection logic
├── chatbot.py              # Chatbot and LLM logic
├── tools/                  # Directory for tools
│   └── youtube_tool.py     # YouTube API tool
└── utils/                  # Utility functions
    ├── voice_input.py      # Voice input logic
    └── text_to_speech.py   # Text-to-speech logic

