from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "ansu0122/docugen"
DATASET_DIR = "dataset"

def upload_dataset_to_hf(repo_id, repo_dir):
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id, exist_ok=True, repo_type="dataset")

    # Ensure dataset folder is a valid Git repo
    os.chdir(repo_dir)
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", f"https://huggingface.co/datasets/{repo_id}"], check=True)

    # Track large files with LFS
    subprocess.run(["git", "lfs", "track", "*.png", "*.jpg", "*.jpeg"], check=True)
    
    # Add and commit changes
    subprocess.run(["git", "add", "--all"], check=True)
    status_output = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout

    if status_output.strip():
        subprocess.run(["git", "commit", "-m", "uploaded dataset"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Dataset uploaded successfully!")
    else:
        print("No changes to commit. Dataset is already up to date.")

if __name__ == "__main__":
    upload_dataset_to_hf(REPO_ID, DATASET_DIR)
