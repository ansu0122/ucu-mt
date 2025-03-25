import pytesseract
import json
import os
import docdataset as dd


def ocr_full_image(image, lang="ukr"):
    try:
        return pytesseract.image_to_string(image, lang=lang).strip()
    except Exception as e:
        print(f"OCR error on full image: {e}")
        return ""


def ocr_cropped_regions(image, grounding, region_types=None, lang="ukr"):
    valid_types = {"text", "table", "chart"}
    if region_types is None or not all(rt in valid_types for rt in region_types):
        raise ValueError("region_types must be a list containing one or more of: 'text', 'table', 'chart'")

    width, height = image.size
    ocr_texts = []

    for region in grounding:
        if region.get("type") not in region_types:
            continue

        box = region["box"]
        left = int(box["l"] * width)
        top = int(box["t"] * height)
        right = int(box["r"] * width)
        bottom = int(box["b"] * height)

        cropped = image.crop((left, top, right, bottom))
        try:
            text = pytesseract.image_to_string(cropped, lang=lang)
            ocr_texts.append(text.strip())
        except Exception as e:
            print(f"OCR error on region: {e}")

    return "\n\n".join(ocr_texts).strip()


def ocr_dataset(dataset, output_jsonl_path, chunk_size=50, lang="ukr", region_types=None):
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # Always overwrite the file at the beginning
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        pass  # clear the file if it exists

    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        chunk = dataset.select(range(start, end))
        print(f"OCR Processing chunk {start+1} to {end}")

        results = []
        for example in chunk:
            img = example["image"]
            grounding = example.get("grounding", [])
            if region_types:
                text = ocr_cropped_regions(img, grounding, region_types=region_types, lang=lang)
            else:
                text = ocr_full_image(img, lang=lang)

            results.append({
                "id": example["id"],
                "text": text
            })

        # Append chunk results to file
        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {len(results)} records to {output_jsonl_path}")


if __name__ == "__main__":
    dataset = dd.download_dataset()['train']
    # ocr_dataset(dataset, "results/ocr_text.jsonl", chunk_size=50, lang="ukr", region_types=["text"])
    ocr_dataset(dataset, "results/ocr_whole_doc.jsonl", chunk_size=50, lang="ukr")
