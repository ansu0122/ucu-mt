import os
import re
import json
from dotenv import load_dotenv
import subprocess
from huggingface_hub import HfApi
from datasets import load_dataset, Image, Dataset
from typing import Optional, List, Dict
import prompt_templates as pt
import string_util as su

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "ansu0122/docugen"
DATASET_DIR = "dataset"
VALID_STYLES = {"print", "hand", "scan"}
VALID_TYPES = {"text", "table", "chart"}

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


def load_results(result_path) -> dict:
    ocr_map = {}
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ocr_map[record["id"]] = record.get("text", "")
    return ocr_map


def validate_filters(style, region_types=None):
    if style is not None:
        if not isinstance(style, list) or not all(s in VALID_STYLES for s in style):
            raise ValueError("style must be a list containing one or more of: 'print', 'hand', 'scan'")

    if region_types is not None:
        if not isinstance(region_types, list) or not all(rt in VALID_TYPES for rt in region_types):
            raise ValueError("region_types must be a list containing one or more of: 'text', 'table', 'chart'")

def _extract_content(entry: Dict, region_types=None) -> Optional[tuple[str, str]]:
    entry_id = entry.get("id")
    if not entry_id:
        return None

    grounding = entry.get("grounding", [])
    text_segments = [
        region["content"].strip()
        for region in grounding
        if region_types is None or region.get("type") in region_types
    ]

    if not text_segments:
        return None

    joined_content = "\n\n".join(text_segments).strip()
    return entry_id, joined_content


def extract_content(
    dataset: Dataset,
    style: Optional[List[str]] = None,
    region_types: Optional[List[str]] = None,
    max_height: int = 1200
) -> Dict[str, str]:

    validate_filters(style, region_types)
    results = {}

    for entry in dataset:
        if style is not None and entry.get("style") not in style:
            continue
        
        image = entry.get("image")
        if image is None:
            return None

        _, height = image.size
        if height > max_height:
            return None

        result = _extract_content(entry, region_types=region_types)
        if result is None:
            continue

        entry_id, content = result
        results[entry_id] = content

    return results


def _extract_titles(entry: Dict, region_types=None) -> Optional[tuple[str, str]]:
    entry_id = entry.get("id")
    if not entry_id:
        return None

    grounding = entry.get("grounding", [])
    titles = []

    for region in grounding:
        if region_types is not None and region.get("type") not in region_types:
            continue

        content = region.get("content", "").strip()
        parts = re.split(r'<table>|\n\n', content, maxsplit=1)
        title = parts[0].strip() if parts else ""
        if title:
            titles.append(title)

    if not titles:
        return None

    return entry_id, "\n".join(titles)


def extract_titles(
    dataset: Dataset,
    style: Optional[List[str]] = None,
    region_types: Optional[List[str]] = None
) -> Dict[str, str]:
    validate_filters(style, region_types)
    results = {}

    for entry in dataset:
        if style is not None and entry.get("style") not in style:
            continue

        result = _extract_titles(entry, region_types=region_types)
        if result is None:
            continue

        entry_id, title_str = result
        results[entry_id] = title_str

    return results


def prep_train_data(dataset: Dataset) -> Dataset:
    results = []

    for entry in dataset:
        image = entry["image"]
        width, height = image.size
        entry_id = entry["id"]

        # --- Full document: text ---
        text_result = _extract_content(entry, region_types=None) # we need all text except tables
        if text_result:
            _, text = text_result
            results.append({
                "id": entry_id,
                "image": image,
                "focus": "doc",
                "content": su.strip_html_table(text),
                "prompt": pt.get_text_template(),
                "requested": "text"
            })

        # --- Full document: table ---
        table_result = _extract_content(entry, region_types=["table"]) # we need only tables
        if table_result:
            _, table = table_result
            results.append({
                "id": entry_id,
                "image": image,
                "focus": "doc",
                "content": su.fetch_html_table(table),
                "prompt": pt.get_table_template(),
                "requested": "table"
            })

        # --- Full document: titles ---
        title_result = _extract_titles(entry, region_types=None)
        if title_result:
            _, titles = title_result
            results.append({
                "id": entry_id,
                "image": image,
                "focus": "doc",
                "content": titles,
                "prompt": pt.get_title_template(),
                "requested": "titles"
            })

        # --- Full document: category classification ---
        results.append({
            "id": entry_id,
            "image": image,
            "focus": "doc",
            "content": entry["category"],
            "prompt": pt.get_class_template(),
            "requested": "category"
        })

        # --- Region-level content ---
        for region in entry.get("grounding", []):
            rtype = region.get("type")
            if rtype not in ["text", "table"]:
                continue

            box = region["box"]
            left = int(box["l"] * width)
            top = int(box["t"] * height)
            right = int(box["r"] * width)
            bottom = int(box["b"] * height)

            cropped = image.crop((left, top, right, bottom))
            content = region["content"]
            formatted_content = (
                su.strip_html_table(content) if rtype == "text" else su.fetch_html_table(content)
            )
            prompt = pt.get_text_template() if rtype == "text" else pt.get_table_template()

            results.append({
                "id": entry_id,
                "image": cropped,
                "focus": rtype,
                "content": formatted_content,
                "prompt": prompt,
                "requested": rtype
                
            })

    return Dataset.from_list(results)


if __name__ == "__main__":
    upload_dataset(REPO_ID, DATASET_DIR)
