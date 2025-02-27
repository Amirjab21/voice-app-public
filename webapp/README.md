# Voice Note Transcriber

A simple web application that allows users to record voice notes and transcribe them to text using OpenAI's Whisper model.

## Features

- Record audio directly from your browser
- Transcribe audio to text using Whisper
- Simple and intuitive user interface
- Supports multiple languages (automatically detected)

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
cd webapp
uvicorn main:app --reload
```

4. Open your browser and navigate to `http://localhost:8000`

## Usage

1. Click the "Start Recording" button to begin recording your voice note
2. Speak clearly into your microphone
3. Click "Stop Recording" when you're finished
4. Wait for the transcription to appear

## Notes

- The application uses the "tiny" Whisper model by default for faster processing
- Make sure your browser has permission to access your microphone
- The audio is processed locally on your machine
- Temporary audio files are automatically cleaned up after processing 