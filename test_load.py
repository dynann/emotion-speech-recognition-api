import json
import os
import sys

# Redirect output to file
log_file = open("debug_output.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

MODEL_PATH = "./wav2vec2-emotion-speech-recognition/checkpoint-2200"

try:
    print(f"Loading feature extractor from {MODEL_PATH}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    print("Feature extractor loaded successfully.")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
    print("Model loaded successfully.")
    print(f"Model config _name_or_path: {model.config._name_or_path}")
    print(f"Model config: {model.config}")
    print(f"Number of labels in config: {model.config.num_labels}")
    print(f"id2label in config: {model.config.id2label}")
    
    # Dummy inference
    import torch
    import numpy as np
    dummy_audio = np.random.uniform(-1, 1, 16000)
    inputs = feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"Inference successful. Logits shape: {logits.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
