import os
from dotenv import load_dotenv
import subprocess
from huggingface_hub import HfApi
from datasets import load_dataset, Image

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "ansu0122/docugen"
DATASET_DIR = "dataset"

def upload_dataset(repo_id, repo_dir):
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id, exist_ok=True, repo_type="dataset")

    os.chdir(repo_dir)
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", f"https://huggingface.co/datasets/{repo_id}"], check=True)

    subprocess.run(["git", "lfs", "track", "*.png", "*.jpg", "*.jpeg"], check=True)
    
    subprocess.run(["git", "add", "--all"], check=True)
    status_output = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout

    if status_output.strip():
        subprocess.run(["git", "commit", "-m", "uploaded dataset"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Dataset uploaded successfully!")
    else:
        print("No changes to commit. Dataset is already up to date.")

def download_dataset():
    dataset = load_dataset(REPO_ID)
    dataset = dataset.cast_column("image", Image())
    return dataset

if __name__ == "__main__":
    upload_dataset(REPO_ID, DATASET_DIR)
