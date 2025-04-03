import json
import os
import docdataset as dd


def default_ocr_fn(ocr_fn):
    def wrapper(image, lang="ukr"):
        try:
            return ocr_fn(image)
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    return wrapper


def ocr_cropped_regions(image, grounding, ocr_fn, region_types=None, lang="ukr"):
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
            text = ocr_fn(cropped)
            ocr_texts.append(text.strip())
        except Exception as e:
            print(f"OCR error on region: {e}")

    return "\n\n".join(ocr_texts).strip()


def ocr_dataset(dataset, output_jsonl_path, ocr_fn, chunk_size=50, lang="ukr", region_types=None):
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        pass

    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        chunk = dataset.select(range(start, end))
        print(f"OCR Processing chunk {start+1} to {end}")

        results = []
        for example in chunk:
            img = example["image"]
            width, height = img.size
            if height > 1200:
                continue
            
            grounding = example.get("grounding", [])
            if region_types:
                text = ocr_cropped_regions(img, grounding, ocr_fn=ocr_fn, region_types=region_types, lang=lang)
            else:
                text = ocr_fn(img)

            results.append({
                "id": example["id"],
                "text": text
            })

        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {len(results)} records to {output_jsonl_path}")


if __name__ == "__main__":
    dataset = dd.download_dataset()['train']

    def pytesseract_ocr(image):
        return pytesseract.image_to_string(image, lang="ukr").strip()

    # OCR cropped regions
    # ocr_dataset(dataset, "results/ocr_text.jsonl", ocr_fn=pytesseract_ocr, chunk_size=50, lang="ukr", region_types=["text"])

    # OCR whole image
    ocr_dataset(dataset, "results/ocr_whole_doc.jsonl", ocr_fn=pytesseract_ocr, chunk_size=50, lang="ukr")
