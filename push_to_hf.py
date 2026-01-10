import os
from huggingface_hub import HfApi, create_repo

# --- Configuration ---
# Replace with your Hugging Face username and desired repo name
REPO_ID = "dynann/emotion-classification" 
LOCAL_FOLDER = "./wav2vec2-emotion-recognition"
# ---------------------

README_CONTENT = f"""
---
license: apache-2.0
datasets:
- stapesai/ssi-speech-emotion-recognition
metrics:
- accuracy
- precision
- recall
- f1
pipeline_tag: audio-classification
---

# Multimodal Emotion Speech Recognition

## Model Description
This model performs emotion recognition from speech using a multimodal approach, utilizing:
- **Audio Model**: Wav2Vec2 Base
## Dataset
- **Dataset Name**: [stapesai/ssi-speech-emotion-recognition](https://huggingface.co/datasets/stapesai/ssi-speech-emotion-recognition)

## Evaluation Results

### Classification Report
```
              precision    recall  f1-score   support

         ANG       0.97      0.93      0.95        30
         CAL       0.00      0.00      0.00         0
         DIS       0.95      0.90      0.92        20
         FEA       0.76      0.70      0.73        27
         HAP       0.87      0.82      0.84        33
         NEU       0.96      0.96      0.96        25
         SAD       0.73      1.00      0.84        19
         SUR       0.88      0.78      0.82         9

    accuracy                           0.87       163
   macro avg       0.76      0.76      0.76       163
weighted avg       0.88      0.87      0.87       163
```

**Overall Accuracy**: 87%
"""

def push_to_hub():
    api = HfApi()

    # 0. Create README.md
    print(f"Creating README.md in {LOCAL_FOLDER}...")
    if not os.path.exists(LOCAL_FOLDER):
        os.makedirs(LOCAL_FOLDER)
    
    with open(os.path.join(LOCAL_FOLDER, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_CONTENT.strip())

    # 1. Create the repository if it doesn't exist
    print(f"Ensuring repository {REPO_ID} exists...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print("Repository is ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        print("Make sure you are logged in using `huggingface-cli login` or have set the HUGGING_FACE_HUB_TOKEN environment variable.")
        return

    # 2. Upload the entire folder
    print(f"Uploading files from {LOCAL_FOLDER} to Hugging Face Hub...")
    try:
        api.upload_folder(
            folder_path=LOCAL_FOLDER,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Initial model upload with model card"
        )
        print(f"Successfully uploaded to https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"Error uploading: {e}")

if __name__ == "__main__":
    # If you haven't logged in via CLI, you can also pass your token directly to HfApi(token="your_token")
    # or set os.environ["HUGGING_FACE_HUB_TOKEN"] = "your_token"
    push_to_hub()

