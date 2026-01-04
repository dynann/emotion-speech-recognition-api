import os
from huggingface_hub import HfApi, create_repo

# --- Configuration ---
# Replace with your Hugging Face username and desired repo name
REPO_ID = "dynann/wav2vec2-emotion-speech-recognition-v4" 
LOCAL_FOLDER = "./wav2vec2-emotion-speech-recognition"
# ---------------------

def push_to_hub():
    api = HfApi()

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
            commit_message="Initial model upload"
        )
        print(f"Successfully uploaded to https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"Error uploading: {e}")

if __name__ == "__main__":
    # If you haven't logged in via CLI, you can also pass your token directly to HfApi(token="your_token")
    # or set os.environ["HUGGING_FACE_HUB_TOKEN"] = "your_token"
    push_to_hub()
