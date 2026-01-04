import json
import torch
import torch.nn as nn
import librosa
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers.modeling_outputs import SequenceClassifierOutput
from huggingface_hub import hf_hub_download
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuration
REPO_ID = "dynann/wav2vec2-emotion-speech-recognition-v4"
# Local fallback or cache path
# MODEL_DIR = "./wav2vec2-emotion-speech-recognition"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 16000
MAX_DURATION = 6.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals
model = None
processor = None
id2label = {}


class Wav2Vec2Emotion(nn.Module):
    def __init__(self, num_labels: int, pretrained: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if attention_mask is not None and hasattr(self.wav2vec, "_get_feature_vector_attention_mask"):
            feat_mask = self.wav2vec._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            if feat_mask.shape[1] == hidden.shape[1]:
                mask = feat_mask.unsqueeze(-1).to(dtype=hidden.dtype)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = (hidden * mask).sum(dim=1) / denom
            else:
                pooled = torch.mean(hidden, dim=1)
        else:
            pooled = torch.mean(hidden, dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_model():
    global model, processor, id2label

    print(f"Loading model resources from {REPO_ID}...")
    
    # 1. Load processor
    processor = Wav2Vec2Processor.from_pretrained(REPO_ID)

    # 2. Download and load labels.json
    labels_path = hf_hub_download(repo_id=REPO_ID, filename="labels.json")
    with open(labels_path, "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # 3. Download and load model.pt (state_dict)
    state_dict_path = hf_hub_download(repo_id=REPO_ID, filename="model.pt")
    
    # Initialize custom model architecture
    model = Wav2Vec2Emotion(num_labels=len(label2id)).to(device)
    state = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("Model + processor loaded successfully from Hugging Face Hub.")


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
        
        # Strategy 1: Try soundfile (fastest for WAV/FLAC)
        try:
            import soundfile as sf
            audio, orig_sr = sf.read(io.BytesIO(file_content))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if orig_sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
        except Exception:
            # Strategy 2: Use ffmpeg via subprocess (handles WebM, MP3, etc.)
            print("Soundfile failed. Attempting ffmpeg...")
            try:
                import subprocess
                # Call ffmpeg to convert input to raw 16k mono 32-bit float PCM
                command = [
                    "ffmpeg",
                    "-i", "pipe:0",          # Read from stdin
                    "-f", "f32le",           # Output format: float 32-bit little endian
                    "-acodec", "pcm_f32le",
                    "-ar", str(SAMPLE_RATE), # 16000
                    "-ac", "1",              # Mono
                    "pipe:1"                 # Write to stdout
                ]
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate(input=file_content)
                
                if process.returncode != 0:
                    raise Exception(f"FFmpeg error: {stderr.decode()}")
                
                audio = np.frombuffer(stdout, dtype=np.float32)
            except Exception as ffmpeg_err:
                print(f"FFmpeg failed: {ffmpeg_err}. Falling back to librosa.load")
                # Strategy 3: librosa.load fallback
                audio, _ = librosa.load(io.BytesIO(file_content), sr=SAMPLE_RATE, mono=True)
            
    except Exception as e:
        print(f"Audio processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")

    try:
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_SAMPLES,
            truncation=True,
        )

        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        results = {id2label.get(i, f"unknown_{i}"): float(p) for i, p in enumerate(probs)}
        sorted_results = dict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))
        top_emotion = next(iter(sorted_results))

        return {"emotion": top_emotion, "others": sorted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")