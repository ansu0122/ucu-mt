import os
import json
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

def extract_gt_content(dataset, style=None, region_types=None) -> list:
    """
    Extract ground-truth content from dataset based on optional style and region type filters.

    Args:
        dataset: Hugging Face Dataset object.
        style (list or None): List of styles to include (e.g. ['print', 'scan']), or None to include all.
        region_types (list): List of region types to include (e.g. ['text', 'table']).

    Returns:
        List of dicts with keys: id, category, style, content
    """
    valid_styles = {"print", "hand", "scan"}
    valid_types = {"text", "table", "chart"}

    if style is not None:
        if not isinstance(style, list) or not all(s in valid_styles for s in style):
            raise ValueError("style must be a list containing one or more of: 'print', 'hand', 'scan'")

    if region_types is not None:
        if not isinstance(region_types, list) or not all(rt in valid_types for rt in region_types):
            raise ValueError("region_types must be a list containing one or more of: 'text', 'table', 'chart'")

    results = {}
    for example in dataset:
        if style is not None and example.get("style") not in style:
            continue

        grounding = example.get("grounding", [])
        text_segments = [
            region["content"].strip()
            for region in grounding
            if region_types is None or region.get("type") in region_types
        ]
        joined_content = "\n\n".join(text_segments).strip()

        results[example["id"]] = joined_content
    return results

import re

def extract_gt_titles(dataset, style=None, region_types=None) -> dict:
    """
    Extract ground-truth titles from dataset based on optional style and region type filters.

    Args:
        dataset: Hugging Face Dataset object.
        style (list or None): List of styles to include (e.g. ['print', 'scan']), or None to include all.
        region_types (list): List of region types to include (e.g. ['text', 'table']).

    Returns:
        Dict mapping id to comma-separated titles string.
    """
    valid_styles = {"print", "hand", "scan"}
    valid_types = {"text", "table", "chart"}

    if style is not None:
        if not isinstance(style, list) or not all(s in valid_styles for s in style):
            raise ValueError("style must be a list containing one or more of: 'print', 'hand', 'scan'")

    if region_types is not None:
        if not isinstance(region_types, list) or not all(rt in valid_types for rt in region_types):
            raise ValueError("region_types must be a list containing one or more of: 'text', 'table', 'chart'")

    results = {}

    for example in dataset:
        if style is not None and example.get("style") not in style:
            continue

        grounding = example.get("grounding", [])
        titles = []

        for region in grounding:
            if region_types is not None and region.get("type") not in region_types:
                continue

            content = region.get("content", "").strip()
            parts = re.split(r'<table>|\n\n', content, maxsplit=1)
            title = parts[0].strip() if parts else ""
            if title:
                titles.append(title)

        if titles:
            results[example["id"]] = ", ".join(titles)

    return results



def load_results(result_path) -> dict:
    ocr_map = {}
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ocr_map[record["id"]] = record.get("text", "")
    return ocr_map



if __name__ == "__main__":
    upload_dataset(REPO_ID, DATASET_DIR)
