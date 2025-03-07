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
import torch
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from whisper.load_model import load_model
from scipy import signal
from accent_model.accent_model import ModifiedWhisper
# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")


lora = True
finetuned_model_path = "model_weights/model_11_accents.pt" #"model_weights/margin0.5-medium-conv2-only.pt" #"model_weights/finetuned-medium-1.pt"
MODEL_VARIANT = "small"
DEVICE = torch.device('cpu')
NUM_ACCENT_CLASSES = 11
ID_TO_ACCENT = {
    0: "Scottish", 1: "English", 2: "Indian", 3: "Irish", 4: "Welsh",
    5: "NewZealandEnglish", 6: "AustralianEnglish", 7: "SouthAfrican",
    8: "Canadian", 9: "NorthernIrish", 10: "American"
}

def setup_model():
    base_whisper_model = load_model(MODEL_VARIANT, device=DEVICE)
    model = ModifiedWhisper(base_whisper_model.dims, NUM_ACCENT_CLASSES, base_whisper_model)
    
    peft_config = LoraConfig(
        inference_mode=True,
        r=8,
        target_modules=["out", "token_embedding", "query", "key", "value", "proj_out"],
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(DEVICE)


    checkpoint = torch.load(finetuned_model_path, map_location=torch.device(DEVICE))
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Model loaded successfully')
        else:
            model.load_state_dict(checkpoint)
            print('Model loaded successfully')
    except Exception as e:
        raise e
    return model, base_whisper_model

model, base_whisper_model = setup_model()


# Initialize FastAPI app
app = FastAPI(title="Whisper Voice Note Transcriber")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# Initialize Whisper model (using the smallest model for quick responses)
# model = whisper.load_model("tiny")

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def display_words_and_probs(result):
    html_output = '<div class="word-prob-container">'
    
    for item in result:
        word = item['word']
        prob = round(item['probability'], 2)
        # Determine color class based on probability
        color_class = 'high-score' if prob >= 0.9 else 'low-score'
        
        html_output += f'''
            <div class="word-prob-pair">
                <div class="word {color_class}">{word}</div>
                <div class="probability {color_class}">{prob:.2f}</div>
            </div>
        '''
    
    html_output += '</div>'
    return html_output

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
                
                audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sampling_rate))
                sampling_rate = 16000
            
            # Process with Whisper
            input_features = processor(
                audio_array, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).input_features
            print("Features extracted successfully")
            print(input_features, temp_file, audio_array, audio_file)

            print(input_features, 'input features')
            output, pooled_embed = model(input_features)
            print(output, 'output')
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # probabilities = torch.nn.functional.softmax(output, dim=1)
            print(probabilities, 'probabilities')
            predictions = torch.argmax(probabilities, dim=1)
            predicted_accent = ID_TO_ACCENT[predictions.item()]
            accent_probabilities = {ID_TO_ACCENT[i]: prob.item() for i, prob in enumerate(probabilities[0])}
            print(wav_path, 'audio file path')

            newloadedmodel = load_model("small", device=DEVICE)
            newloadedmodel.to(DEVICE)
            outputtemp = newloadedmodel.transcribe(str(wav_path), word_timestamps=True, language="en")
            print(outputtemp, 'outputtemp')
            output = base_whisper_model.transcribe(str(wav_path), word_timestamps=True, language="en")
            print(output, 'output')
            relevant_info = outputtemp['segments'][0]['words']
            innerHTML = display_words_and_probs(relevant_info)
            # innerHTML = ""

            # Create data for the pie chart
            pie_chart_data = {
                'labels': list(accent_probabilities.keys()),
                'values': list(accent_probabilities.values())
            }

            return {
                "success": True,
                "text": relevant_info[0]['word'], # output['text'],
                "innerHTML": innerHTML,
                "language": "English",
                "predicted_accent": predicted_accent,
                "accent_probabilities": pie_chart_data
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




@app.post("/placeholder/")
async def placeholder(request: Request):
    try:
        body = await request.json()
        audio_file_path = body.get('audioFile', "./sample_audio/BI0003_scottish.wav")
        
        audio_array, sampling_rate = sf.read(audio_file_path)
        print(f"Audio loaded successfully: shape={audio_array.shape}, sampling_rate={sampling_rate}")
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sampling_rate != 16000:
            audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sampling_rate))
            sampling_rate = 16000
        
        # Process with Whisper
        input_features = processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features
        print("Features extracted successfully")

        print(input_features, 'input features')
        output, pooled_embed = model(input_features)
        print(output, 'output')
        probabilities = torch.nn.functional.softmax(output, dim=1)
        print(probabilities, 'probabilities')
        predictions = torch.argmax(probabilities, dim=1)
        predicted_accent = ID_TO_ACCENT[predictions.item()]
        accent_probabilities = {ID_TO_ACCENT[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        print(audio_file_path, 'audio file path')

        newloadedmodel = load_model("small", device=DEVICE)
        newloadedmodel.to(DEVICE)
        outputtemp = newloadedmodel.transcribe(str(audio_file_path), word_timestamps=True, language="en")
        print(outputtemp, 'outputtemp')
        output = base_whisper_model.transcribe(str(audio_file_path), word_timestamps=True, language="en")
        print(output, 'output')
        relevant_info = outputtemp['segments'][0]['words']
        innerHTML = display_words_and_probs(relevant_info)
        # innerHTML = ""

        # Create data for the pie chart
        pie_chart_data = {
            'labels': list(accent_probabilities.keys()),
            'values': list(accent_probabilities.values())
        }

        return {
            "success": True,
            "text": relevant_info[0]['word'], # output['text'],
            "innerHTML": innerHTML,
            "language": "English",
            "predicted_accent": predicted_accent,
            "accent_probabilities": pie_chart_data
        }
    except IOError as e:
        print(f"IOError while processing file: {str(e)}")
        return {"success": False, "error": f"Failed to process file: {str(e)}"}
    
    
        