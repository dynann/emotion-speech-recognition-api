import requests
import numpy as np
import io
import soundfile as sf

def test_api():
    url = "http://127.0.0.1:8000/generate-emotion"
    
    # Create a dummy 1-second sine wave
    sr = 16000
    t = np.linspace(0, 1, sr)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Save to a buffer as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    
    print("Sending request to API...")
    try:
        files = {'file': ('test.wav', buffer, 'audio/wav')}
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
