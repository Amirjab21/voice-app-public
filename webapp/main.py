import os
from pathlib import Path
from typing import Optional

import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
from pydub import AudioSegment
import io

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model = whisper.load_model("small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="translate")

# load streaming dataset and read first audio sample
# ds = load_dataset("common_voice", "Persian", split="test", streaming=True)
# ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
# input_speech = next(iter(ds))["audio"]
# input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

# # generate token ids
# predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
# # decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# [' A very interesting work, we will finally be given on this subject.']

# Initialize FastAPI app
app = FastAPI(title="Whisper Voice Note Transcriber")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize Whisper model (using the smallest model for quick responses)
# model = whisper.load_model("tiny")

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    print("Transcribing audio started")
    # Save the uploaded file temporarily
    temp_file = UPLOAD_DIR / audio_file.filename
    print(f"Temp file path: {temp_file}")
    print(f"Upload directory exists: {UPLOAD_DIR.exists()}")
    print(f"Upload directory is writable: {os.access(UPLOAD_DIR, os.W_OK)}")
    
    try:
        print(f"Reading file content, content_type: {audio_file.content_type}")
        contents = await audio_file.read()
        print(f"Content length: {len(contents)} bytes")
        
        try:
            # Save the webm file
            with open(temp_file, "wb") as f:
                bytes_written = f.write(contents)
                print(f"Bytes written to file: {bytes_written}")
            
            # Convert webm to wav using pydub
            audio = AudioSegment.from_file(temp_file, format="webm")
            wav_path = temp_file.with_suffix('.wav')
            audio.export(wav_path, format="wav")
            print("Converted to WAV format successfully")
            
            # Load the WAV file
            audio_array, sampling_rate = sf.read(wav_path)
            print(f"Audio loaded successfully: shape={audio_array.shape}, sampling_rate={sampling_rate}")
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                from scipy import signal
                audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sampling_rate))
                sampling_rate = 16000
            
            # Process with Whisper
            input_features = processor(
                audio_array, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).input_features
            print("Features extracted successfully")
            
            # Generate token ids
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            # Decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return {
                "success": True,
                "text": transcription,
                "language": "English"  # Since we're using French translation
            }
        except IOError as e:
            print(f"IOError while processing file: {str(e)}")
            return {"success": False, "error": f"Failed to process file: {str(e)}"}
            
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        # Clean up the temporary files
        try:
            if temp_file.exists():
                temp_file.unlink()
            wav_path = temp_file.with_suffix('.wav')
            if wav_path.exists():
                wav_path.unlink()
            print("Temporary files cleaned up successfully")
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}") 


def display_words_and_probs(result):
    # Start with container div
    html_output = '<div class="word-prob-container">'
    
    # Create individual word-probability pairs in their own divs
    for item in result:
        word = item['word']
        prob = round(item['probability'], 2)
        html_output += f'''
            <div class="word-prob-pair">
                <div class="word">{word}</div>
                <div class="probability">{prob:.2f}</div>
            </div>
        '''
    
    html_output += '</div>'
    return html_output

@app.post("/placeholder/")
async def placeholder():
    try:

        audio_file = "../hello.wav" #change from hardcoded to uploaded file
        output = model.transcribe(audio_file, word_timestamps=True, language="en")
        print(output)
        print(output['segments'][0]['words'])
        relevant_info = output['segments'][0]['words']
        print(relevant_info)
        innerHTML = display_words_and_probs(relevant_info)
        return {
            "success": True,
            "text": relevant_info[0]['word'],
            "language": "English",
            "innerHTML": innerHTML
        }
    except IOError as e:
        print(f"IOError while processing file: {str(e)}")
        return {"success": False, "error": f"Failed to process file: {str(e)}"}
        