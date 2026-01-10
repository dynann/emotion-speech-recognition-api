import json
import torch
import torch.nn as nn
import librosa
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    RobertaModel,
    AutoTokenizer,
    pipeline,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from huggingface_hub import hf_hub_download
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- CONFIG ---
REPO_ID = "dynann/emotion-classification"
SAMPLE_RATE = 16000
MAX_DURATION = 6.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals
model = None
processor = None
id2label = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODEL ARCHITECTURE (Updated to Speech-Only) ---
class SpeechEmotionModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        # Audio Encoder
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Classifier (Audio dim 768 -> 512 -> num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_values, audio_attention_mask=None, labels=None):
        # Audio forward
        audio_outputs = self.wav2vec(input_values, attention_mask=audio_attention_mask)

        # Mean Pooling for Audio features
        if audio_attention_mask is not None:
            # Handle variable length audio mask
            scale = (
                audio_outputs.last_hidden_state.shape[1] / audio_attention_mask.shape[1]
            )
            feat_mask_len = (audio_attention_mask.sum(dim=1) * scale).long()
            feat_mask = torch.zeros(
                audio_outputs.last_hidden_state.shape[:2],
                device=audio_outputs.last_hidden_state.device,
            )
            for i, l in enumerate(feat_mask_len):
                feat_mask[i, :l] = 1

            mask = feat_mask.unsqueeze(-1).to(
                dtype=audio_outputs.last_hidden_state.dtype
            )
            denom = mask.sum(dim=1).clamp(min=1.0)
            audio_features = (audio_outputs.last_hidden_state * mask).sum(dim=1) / denom
        else:
            audio_features = torch.mean(audio_outputs.last_hidden_state, dim=1)

        logits = self.classifier(audio_features)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# --- LOADING LOGIC ---
def load_model():
    global model, processor, id2label

    print(f"Loading speech emotion model resources from {REPO_ID}...")

    # 1. Load processor (audio only)
    # This automatically downloads from the HF Hub
    processor = Wav2Vec2Processor.from_pretrained(REPO_ID)

    # 2. Download and load labels.json from Hub
    labels_path = hf_hub_download(repo_id=REPO_ID, filename="labels.json")
    with open(labels_path, "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # 3. Download and load model.pt (state_dict) from Hub
    state_dict_path = hf_hub_download(repo_id=REPO_ID, filename="model.pt")

    # Initialize the speech-only architecture
    model = SpeechEmotionModel(num_labels=len(label2id)).to(device)
    state = torch.load(state_dict_path, map_location=device)

    # Load weights
    model.load_state_dict(state)
    model.eval()

    print(f"Speech model loaded successfully with {len(label2id)} emotions.")


# Load once at startup
try:
    load_model()
except Exception as e:
    print(f"Error loading model resources: {e}")
    model = None
    processor = None
    id2label = {}


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Emotion Recognition API is running"}


@app.post("/generate-emotion")
async def generate_emotion(file: UploadFile = File(...)):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model currently unavailable")

    try:
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        print(f"Received file: {file.filename}, size: {len(file_content)} bytes")

        # Strategy 1: Try soundfile (fastest for WAV/FLAC)
        try:
            import soundfile as sf
            audio, orig_sr = sf.read(io.BytesIO(file_content))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if orig_sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
            print("Successfully loaded audio via soundfile")
        except Exception as sf_err:
            # Strategy 2: Use ffmpeg via subprocess (handles WebM, MP3, etc.)
            print(f"Soundfile failed: {sf_err}. Attempting ffmpeg...")
            try:
                import subprocess
                import shutil

                ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg"
                
                # Call ffmpeg to convert input to raw 16k mono 32-bit float PCM
                command = [
                    ffmpeg_path,
                    "-i", "pipe:0",  # Read from stdin
                    "-f", "f32le",   # Output format: float 32-bit little endian
                    "-acodec", "pcm_f32le",
                    "-ar", str(SAMPLE_RATE),  # 16000
                    "-ac", "1",      # Mono
                    "-y",            # Overwrite output
                    "pipe:1",        # Write to stdout
                ]
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = process.communicate(input=file_content)

                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    raise Exception(f"FFmpeg error: {error_msg}")

                audio = np.frombuffer(stdout, dtype=np.float32)
                if len(audio) == 0:
                    raise Exception("FFmpeg returned empty audio data")
                print("Successfully loaded audio via ffmpeg")
            except Exception as ffmpeg_err:
                print(f"FFmpeg failed: {ffmpeg_err}. Falling back to librosa.load")
                # Strategy 3: librosa.load fallback
                try:
                    # Strategy 3: librosa.load with temporary file (more robust for some backends)
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                        tmp.write(file_content)
                        tmp_path = tmp.name
                    
                    try:
                        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                        print("Successfully loaded audio via librosa (temp file)")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                except Exception as librosa_err:
                    print(f"Librosa failed: {librosa_err}")
                    raise Exception(f"All audio loading strategies failed.\n"
                                    f"1. Soundfile: {sf_err}\n"
                                    f"2. FFmpeg: {ffmpeg_err}\n"
                                    f"3. Librosa: {librosa_err}")

    except Exception as e:
        print(f"Audio processing error: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error processing audio file: {str(e)}"
        )

    try:
        # ========================================
        # MATCH TRAINING PREPROCESSING EXACTLY
        # ========================================
        
        # 1. Manual truncation (EXACTLY like training)
        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        
        # 2. Process WITHOUT padding/truncation params (EXACTLY like training)
        audio_inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        
        input_values = audio_inputs["input_values"].to(device)
        
        # 3. Handle attention mask (EXACTLY like training)
        if hasattr(audio_inputs, "attention_mask") and audio_inputs.attention_mask is not None:
            audio_attention_mask = audio_inputs.attention_mask.to(device)
        else:
            audio_attention_mask = torch.ones(input_values.shape, dtype=torch.long, device=device)

        # 4. Inference
        with torch.no_grad():
            outputs = model(
                input_values=input_values,
                audio_attention_mask=audio_attention_mask,
            )
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        results = {
            id2label.get(i, f"unknown_{i}"): float(p) for i, p in enumerate(probs)
        }
        sorted_results = dict(
            sorted(results.items(), key=lambda kv: kv[1], reverse=True)
        )
        top_emotion = next(iter(sorted_results))

        return {
            "emotion": top_emotion,
            "transcription": "N/A (Speech-Only model)",
            "others": sorted_results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")