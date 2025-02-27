# Author: Fatih E. NAR
# This is a Voice-to-text GenAI ChatBot Web App.
# In order to use this app you need to deploy a model on https://maas.apps.prod.rhoai.rh-aiservices-bu.com/ and retrieve the API Key
#
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import speech_recognition as sr
from gtts import gTTS
import requests
import os
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("voicebot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# API Configuration - Define constants at the top level for easy modification
API_BASE_URL = 'https://mistral-7b-instruct-v0-3-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443'
API_ENDPOINT = '/v1/completions'
# Get API key from environment variable or use default value
API_KEY = 'api_key' # <<<------------------- Your MaaS Key Goes Here!
MODEL_NAME = "mistral-7b-instruct"
MAX_CONTEXT_LENGTH = 6000
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
REQUEST_TIMEOUT = 45  # seconds

# Create a directory for static files if it doesn't exist
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Conversation history
conversation_history = []

# HTML content as a string with improved UI
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telco Customer Support Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #0057b8;
            --primary-light: #3a7ac5;
            --primary-dark: #003c7e;
            --secondary-color: #f9f9f9;
            --text-color: #333;
            --light-gray: #f2f2f2;
            --border-color: #e0e0e0;
            --success-color: #34c759;
            --error-color: #ff3b30;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --animation-speed: 0.3s;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--secondary-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 2rem;
        }
        
        header p {
            color: #666;
            font-weight: 300;
        }
        
        .chat-container {
            border-radius: 16px;
            overflow: hidden;
            background-color: white;
            box-shadow: 0 8px 24px var(--shadow-color);
            height: calc(100vh - 140px);
            min-height: 500px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-header {
            background-color: var(--primary-color);
            color: white;
            padding: 15px 20px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .chat-header svg {
            width: 24px;
            height: 24px;
        }
        
        .chat-body {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        
        .chat-box {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            scroll-behavior: smooth;
        }
        
        .message {
            margin-bottom: 16px;
            padding: 12px 16px;
            border-radius: 16px;
            max-width: 80%;
            word-wrap: break-word;
            position: relative;
            animation: fadeIn var(--animation-speed) ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background-color: var(--primary-color);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
            box-shadow: 0 2px 6px rgba(0, 87, 184, 0.2);
        }
        
        .ai-message {
            background-color: var(--light-gray);
            color: var(--text-color);
            margin-right: auto;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 6px var(--shadow-color);
        }
        
        .message-time {
            font-size: 0.7rem;
            margin-top: 6px;
            opacity: 0.8;
            text-align: right;
        }
        
        .status-bar {
            background-color: #fff;
            padding: 10px 20px;
            border-top: 1px solid var(--border-color);
            font-size: 0.85rem;
            color: #666;
            transition: all 0.3s ease;
        }
        
        .status-bar.error {
            background-color: rgba(255, 59, 48, 0.1);
            color: var(--error-color);
        }
        
        .status-bar.success {
            background-color: rgba(52, 199, 89, 0.1);
            color: var(--success-color);
        }
        
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 4px;
            margin: 0 0 16px 16px;
            padding: 10px 16px;
            background-color: var(--light-gray);
            border-radius: 16px;
            border-bottom-left-radius: 4px;
            width: fit-content;
        }

        .typing-indicator span {
            height: 8px;
            width: 8px;
            background: var(--primary-color);
            border-radius: 50%;
            display: block;
            opacity: 0.6;
        }

        .typing-indicator span:nth-child(1) {
            animation: bounce 1s infinite 0.1s;
        }
        .typing-indicator span:nth-child(2) {
            animation: bounce 1s infinite 0.2s;
        }
        .typing-indicator span:nth-child(3) {
            animation: bounce 1s infinite 0.3s;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .input-area {
            padding: 16px;
            background-color: white;
            border-top: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .message-input {
            flex: 1;
            position: relative;
        }
        
        .message-input input {
            width: 100%;
            padding: 14px 20px;
            border: 1px solid var(--border-color);
            border-radius: 24px;
            font-family: inherit;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        
        .message-input input:focus {
            border-color: var(--primary-light);
            box-shadow: 0 0 0 3px rgba(0, 87, 184, 0.1);
        }
        
        .action-buttons {
            display: flex;
            gap: 8px;
        }
        
        .action-button {
            background-color: transparent;
            border: none;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            color: var(--primary-color);
            transition: background-color 0.2s ease, transform 0.2s ease;
        }
        
        .action-button:hover {
            background-color: rgba(0, 87, 184, 0.1);
        }
        
        .action-button:active {
            transform: scale(0.95);
        }
        
        .action-button svg {
            width: 24px;
            height: 24px;
            fill: currentColor;
        }
        
        .action-button.recording {
            color: var(--error-color);
            animation: pulse 1.5s infinite;
        }
        
        .action-button.recording svg {
            fill: var(--error-color);
        }
        
        .action-button.recording:hover {
            background-color: rgba(255, 59, 48, 0.1);
        }
        
        .send-button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.2s ease;
        }
        
        .send-button:hover {
            background-color: var(--primary-light);
        }
        
        .send-button:active {
            transform: scale(0.95);
        }
        
        .send-button svg {
            width: 20px;
            height: 20px;
            fill: currentColor;
        }
        
        .audio-player {
            width: 100%;
            margin: 8px 0 0;
            display: none;
            border-radius: 8px;
            height: 36px;
        }
        
        @media (max-width: 600px) {
            body {
                padding: 12px;
            }
            
            .chat-container {
                height: calc(100vh - 110px);
                min-height: 400px;
            }
            
            .message {
                max-width: 85%;
            }
            
            .message-input input {
                padding: 12px 16px;
                font-size: 0.95rem;
            }
            
            .action-button, .send-button {
                width: 40px;
                height: 40px;
            }
        }

        /* Floating scroll-to-bottom button */
        .scroll-bottom {
            position: absolute;
            bottom: 16px;
            right: 16px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: white;
            box-shadow: 0 2px 8px var(--shadow-color);
            display: none;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 10;
            transition: all 0.2s ease;
        }
        
        .scroll-bottom:hover {
            background-color: var(--light-gray);
        }
        
        .scroll-bottom svg {
            width: 20px;
            height: 20px;
            fill: var(--text-color);
        }
    </style>
</head>
<body>
    <header>
        <h1>Telco Customer Support</h1>
        <p>Ask a question or describe your issue. Type or speak.</p>
    </header>
    
    <div class="chat-container">
        <div class="chat-header">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <span>Customer Support Conversation</span>
        </div>
        
        <div class="chat-body">
            <div class="chat-box" id="chat-box">
                <!-- Chat messages will appear here -->
                <div class="ai-message message">
                    Hello! I'm your AI assistant. How can I help you today?
                    <div class="message-time">Now</div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            
            <div class="scroll-bottom" id="scroll-bottom">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </div>
        </div>
        
        <div class="status-bar" id="status">Ready to assist you</div>
        
        <div class="input-area">
            <div class="message-input">
                <input type="text" id="text-input" placeholder="Type your message..." autocomplete="off">
            </div>
            
            <div class="action-buttons">
                <button id="record-button" class="action-button" title="Record voice message">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="23"></line>
                        <line x1="8" y1="23" x2="16" y2="23"></line>
                    </svg>
                </button>
                
                <button id="clear-button" class="action-button" title="Clear conversation">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
                
                <button id="send-button" class="send-button" title="Send message">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
        </div>
    </div>
    
    <audio id="response-audio" class="audio-player" controls></audio>

    <script>
        // DOM elements
        const chatBox = document.getElementById('chat-box');
        const textInput = document.getElementById('text-input');
        const sendButton = document.getElementById('send-button');
        const recordButton = document.getElementById('record-button');
        const clearButton = document.getElementById('clear-button');
        const scrollBottomButton = document.getElementById('scroll-bottom');
        const statusDiv = document.getElementById('status');
        const audioElement = document.getElementById('response-audio');
        const typingIndicator = document.getElementById('typing-indicator');
        
        // State variables
        let isRecording = false;
        let isScrolledToBottom = true;
        
        // Initialize
        textInput.focus();
        
        // Event listeners
        textInput.addEventListener('keypress', handleKeyPress);
        sendButton.addEventListener('click', sendText);
        recordButton.addEventListener('click', toggleRecording);
        clearButton.addEventListener('click', clearChat);
        chatBox.addEventListener('scroll', checkScrollPosition);
        scrollBottomButton.addEventListener('click', scrollToBottom);
        
        // Check if we should show the scroll-to-bottom button
        function checkScrollPosition() {
            const scrollPosition = chatBox.scrollTop + chatBox.clientHeight;
            const scrollHeight = chatBox.scrollHeight;
            
            isScrolledToBottom = Math.abs(scrollHeight - scrollPosition) < 50;
            
            if (isScrolledToBottom) {
                scrollBottomButton.style.display = 'none';
            } else {
                scrollBottomButton.style.display = 'flex';
            }
        }
        
        // Scroll to the bottom of the chat
        function scrollToBottom() {
            chatBox.scrollTop = chatBox.scrollHeight;
            checkScrollPosition();
        }
        
        // Update status message
        function updateStatus(message, type = '') {
            statusDiv.textContent = message;
            statusDiv.className = 'status-bar';
            
            if (type) {
                statusDiv.classList.add(type);
            }
            
            // Reset status after 5 seconds
            if (type === 'success' || type === 'error') {
                setTimeout(() => {
                    statusDiv.className = 'status-bar';
                    statusDiv.textContent = 'Ready to assist you';
                }, 5000);
            }
        }
        
        // Toggle recording state
        function toggleRecording() {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        }
        
        // Start voice recording
        function startRecording() {
            isRecording = true;
            recordButton.classList.add('recording');
            updateStatus('Listening... Speak now');
            
            // Disable text input while recording
            textInput.disabled = true;
            
            fetch('/api/record_voice', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateStatus(data.error, 'error');
                    return;
                }
                
                addMessage(data.user_message, 'user');
                showTypingIndicator();
                
                // Slight delay to make it feel more natural
                setTimeout(() => {
                    hideTypingIndicator();
                    addMessage(data.response, 'ai');
                    
                    // Play the AI's response as audio
                    audioElement.src = `/static/response_${data.timestamp}.mp3`;
                    audioElement.style.display = 'block';
                    audioElement.play();
                    
                    updateStatus('Response received', 'success');
                }, 800);
            })
            .catch(error => {
                updateStatus('Could not process your request. Please try again.', 'error');
                console.error('Error:', error);
            })
            .finally(() => {
                stopRecording();
                // Re-enable text input
                textInput.disabled = false;
                textInput.focus();
            });
        }
        
        // Stop voice recording
        function stopRecording() {
            isRecording = false;
            recordButton.classList.remove('recording');
        }
        
        // Add a message to the chat
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `${sender}-message message`;
            
            const now = new Date();
            const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            messageDiv.innerHTML = `
                ${text}
                <div class="message-time">${timeString}</div>
            `;
            
            chatBox.appendChild(messageDiv);
            scrollToBottom();
        }
        
        // Clear the chat history
        function clearChat() {
            // Confirm before clearing
            if (chatBox.childNodes.length <= 1 || confirm('Are you sure you want to clear the conversation?')) {
                // Keep the first AI greeting message
                while (chatBox.childNodes.length > 1) {
                    chatBox.removeChild(chatBox.lastChild);
                }
                
                audioElement.style.display = 'none';
                audioElement.pause();
                
                fetch('/api/clear_history', { method: 'POST' })
                    .then(response => response.json())
                    .then(() => {
                        updateStatus('Conversation cleared', 'success');
                    });
            }
        }
        
        // Show typing indicator
        function showTypingIndicator() {
            typingIndicator.style.display = 'flex';
            scrollToBottom();
        }
        
        // Hide typing indicator
        function hideTypingIndicator() {
            typingIndicator.style.display = 'none';
        }

        // Send text message
        function sendText() {
            const message = textInput.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            showTypingIndicator();
            textInput.value = '';
            
            fetch('/api/send_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateStatus(data.error, 'error');
                    return;
                }
                
                setTimeout(() => {
                    hideTypingIndicator();
                    addMessage(data.response, 'ai');
                    
                    // Play the AI's response as audio
                    audioElement.src = `/static/response_${data.timestamp}.mp3`;
                    audioElement.style.display = 'block';
                    audioElement.play();
                    
                    updateStatus('Response received', 'success');
                }, 800);
            })
            .catch(error => {
                hideTypingIndicator();
                updateStatus('Could not process your request', 'error');
                console.error('Error:', error);
            });
        }
        
        // Handle Enter key press in the input field
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendText();
            }
        }
        
        // Initialize UI
        scrollToBottom();
    </script>
</body>
</html>
"""
 
def get_api_headers():
    """Return properly formatted API headers with authentication."""
    return {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_CONTENT)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)

@app.route('/api/record_voice', methods=['POST'])
def record_voice():
    """Record user's voice and get AI response."""
    try:
        user_message = recognize_speech()
        timestamp = int(time.time())
        
        if user_message:
            logger.info(f"Recognized text: {user_message}")
            conversation_history.append({"role": "user", "content": user_message})
            
            response_text = get_model_response(user_message)
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Generate a unique filename with timestamp
            audio_filename = f"response_{timestamp}.mp3"
            text_to_speech(response_text, audio_filename)
            
            return jsonify({
                "user_message": user_message, 
                "response": response_text,
                "timestamp": timestamp
            })
        else:
            return jsonify({
                "error": "Sorry, I could not understand the audio. Please try again."
            })
    except Exception as e:
        logger.error(f"Error in voice recording: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your voice. Please try again."
        })

@app.route('/api/send_text', methods=['POST'])
def send_text():
    """Process text input from the user."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        timestamp = int(time.time())
        
        if user_message:
            logger.info(f"Text input: {user_message}")
            conversation_history.append({"role": "user", "content": user_message})
            
            response_text = get_model_response(user_message)
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Generate a unique filename with timestamp
            audio_filename = f"response_{timestamp}.mp3"
            text_to_speech(response_text, audio_filename)
            
            return jsonify({
                "response": response_text,
                "timestamp": timestamp
            })
        else:
            return jsonify({
                "error": "Please enter a message."
            })
    except Exception as e:
        logger.error(f"Error in text processing: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your message. Please try again."
        })

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear the conversation history."""
    global conversation_history
    conversation_history = []
    return jsonify({"status": "success"})

def recognize_speech():
    """Record and recognize speech from the microphone."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logger.info("Listening for speech...")
            # Adjust for ambient noise and set dynamic energy threshold
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = 300  # Adjust based on your environment
            
            # Listen for speech with timeout
            audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=15.0)
            
            logger.info("Speech recorded, recognizing...")
            try:
                # Use Google's speech recognition service
                text = recognizer.recognize_google(audio)
                logger.info(f"Recognized: {text}")
                return text
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                return None
    except Exception as e:
        logger.error(f"Error in speech recognition: {e}")
        return None

def test_api_connection():
    """Test the connection to the API endpoint and print detailed information."""
    logger.info(f"Testing connection to: {API_BASE_URL}{API_ENDPOINT}")
    
    # Get headers from the common function
    headers = get_api_headers()
    
    # Simple test message with required parameters
    data = {
        "model": MODEL_NAME, 
        "prompt": "Hello, are you working?",
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        # Make the request with extended timeout and verbose error reporting
        response = requests.post(
            f"{API_BASE_URL}{API_ENDPOINT}", 
            headers=headers, 
            json=data, 
            timeout=REQUEST_TIMEOUT
        )
        
        # Print full response details for debugging
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response Headers: {response.headers}")
        logger.info(f"Response Content: {response.text[:500]}...")  # Print first 500 chars
        
        # Try to parse as JSON and handle failures gracefully
        try:
            json_response = response.json()
            logger.info("JSON Response parsed successfully")
            return True
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response content: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False

def get_model_response(user_message):
    """Get response from the LLM API with improved error handling and retries."""
    # Calculate the length of the user message
    message_length = len(user_message.split())
    max_tokens = max(0, (MAX_CONTEXT_LENGTH - message_length - 10))
    
    # Get headers from the common function
    headers = get_api_headers()

    # Format for completions API instead of chat API
    system_prompt = "You are an AT&T customer support representative. The following is a customer query, please respond politely and helpfully."
    
    # Build conversation context
    conversation_context = system_prompt + "\n\n"
    
    # Add conversation history (limited to last 10 exchanges to keep context manageable)
    if conversation_history:
        for msg in conversation_history[-10:]:
            role = "Customer: " if msg.get("role") == "user" else "Support: "
            conversation_context += role + msg.get("content", "") + "\n"
    
    # Add current message if not already in history
    if not conversation_history or conversation_history[-1]["content"] != user_message:
        conversation_context += "Customer: " + user_message + "\n"
    
    # Add a prompt for the response
    conversation_context += "Support: "

    # Use the format from the example docs with required model parameter
    data = {
        "model": MODEL_NAME,
        "prompt": conversation_context,
        "max_tokens": max_tokens, 
        "temperature": 0.7
    }

    # Implement retry logic
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Making API request (attempt {attempt+1}/{MAX_RETRIES})")
            
            # Make the request with timeout
            response = requests.post(
                f"{API_BASE_URL}{API_ENDPOINT}", 
                headers=headers, 
                json=data, 
                timeout=REQUEST_TIMEOUT
            )
            
            # Check for successful status code
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    # Extract text from completions API format (different from chat format)
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        if 'text' in response_data['choices'][0]:
                            ai_response = response_data['choices'][0]['text']
                            logger.info("API response received successfully")
                            return ai_response
                        elif 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                            ai_response = response_data['choices'][0]['message']['content']
                            logger.info("API response received successfully (chat format)")
                            return ai_response
                    
                    logger.error(f"Unexpected response structure")
                    logger.error(f"Response data: {response_data}")
                    return "I'm experiencing technical difficulties understanding the API response structure."
                except (KeyError, IndexError) as e:
                    logger.error(f"Unexpected response structure: {e}")
                    logger.error(f"Response data: {response_data}")
                    return "I'm experiencing technical difficulties understanding the API response structure."
                except requests.exceptions.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response: {response.text[:500]}")
                    return "I'm having trouble parsing the API response. Please try again later."
            else:
                logger.error(f"API returned error status: {response.status_code}")
                logger.error(f"Response content: {response.text[:500]}")
                
                # If rate limited, wait longer before retrying
                if response.status_code == 429:
                    time.sleep(RETRY_DELAY * (attempt + 2))  # Progressive backoff
                    continue
                    
            # If we made it here, something went wrong with the API but we have a response
            # Extract any helpful text from the response for debugging
            error_info = ""
            try:
                if response.text and len(response.text) > 0:
                    error_info = f" Response: {response.text[:100]}..."
            except:
                pass
                
            logger.warning(f"API request failed on attempt {attempt+1}{error_info}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to API endpoint (attempt {attempt+1})")
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out (attempt {attempt+1})")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e} (attempt {attempt+1})")
        
        # Wait before retrying (unless this was the last attempt)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    # If we've exhausted all retries, return a fallback response
    return "I'm sorry, I'm having trouble connecting to my AI service right now. Please try again later or contact customer support at 1-800-HELP-ATT for assistance."

def text_to_speech(text, filename="response.mp3"):
    """Convert text to speech and save as an MP3 file."""
    try:
        full_path = os.path.join(STATIC_DIR, filename)
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(full_path)
        logger.info(f"Speech saved to {full_path}")
        return full_path
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {e}")
        # Create a simple error message as speech
        try:
            error_tts = gTTS(text="Sorry, I couldn't generate speech for that response.", lang='en')
            error_tts.save(os.path.join(STATIC_DIR, filename))
        except:
            pass
        return None

if __name__ == '__main__':
    try:
        # Print startup information
        logger.info("="*50)
        logger.info("Starting Telco CRM VoiceBot")
        logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"API URL: {API_BASE_URL}{API_ENDPOINT}")
        logger.info(f"Static directory: {STATIC_DIR}")
        
        # Test the API connection before starting the server
        logger.info("Testing API connection...")
        connection_successful = test_api_connection()
        
        if not connection_successful:
            logger.warning("API connection test failed! The app may not function correctly.")
            logger.warning("Please check your API_URL and API_KEY values.")
        else:
            logger.info("API connection test successful!")
        
        # Create a welcome message TTS file
        welcome_text = "Welcome to the AT&T Customer Support. I'm your AI assistant. How can I help you today?"
        text_to_speech(welcome_text, "welcome.mp3")
        
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=15000, debug=False)
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
