import docdataset as dd
import docmeasure as dm

def remove_table_from_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Skip lines that look like table rows (contain at least 2 '|' characters)
        if line.count('|') >= 2:
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()

import re

def remove_html_tables(text: str) -> str:
    # Remove <table>...</table> blocks (non-greedy match)
    return re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

def keep_only_html_tables(text: str) -> str:
    # Find all <table>...</table> blocks
    tables = re.findall(r'<table.*?>.*?</table>', text, flags=re.DOTALL | re.IGNORECASE)
    return '\n\n'.join(tables).strip()

def ocr_benchmarking(dataset, prediction_path, region_types=['text', 'table']):
    prediction = dd.load_results(prediction_path)
    prediction = {key : remove_table_from_text(item) for key, item in prediction.items()}

    ground_truth = dd.extract_gt_content(dataset, region_types=region_types)
    ground_truth = {key : remove_html_tables(item) for key, item in ground_truth.items() if key in prediction}

    mean_cer = dm.mean_cer(ground_truth, prediction)
    mean_wer = dm.mean_wer(ground_truth, prediction)
    print(f"Mean CER: {mean_cer:.4f} Mean WER: {mean_wer:.4f}")

    ground_truth_print = dd.extract_gt_content(dataset, style=['print'], region_types=region_types)
    ground_truth_print = {key : remove_html_tables(item) for key, item in ground_truth_print.items() if key in prediction}
    mean_cer_print = dm.mean_cer(ground_truth_print, prediction)
    mean_wer_print = dm.mean_wer(ground_truth_print, prediction)
    print(f"Mean CER print: {mean_cer_print:.4f} Mean WER print: {mean_wer_print:.4f}")

    ground_truth_hand = dd.extract_gt_content(dataset, style=['hand'], region_types=region_types)
    ground_truth_hand = {key : remove_html_tables(item) for key, item in ground_truth_hand.items() if key in prediction}
    mean_cer_hand = dm.mean_cer(ground_truth_hand, prediction)
    mean_wer_hand = dm.mean_wer(ground_truth_hand, prediction)
    print(f"Mean CER hand: {mean_cer_hand:.4f} Mean WER hand: {mean_wer_hand:.4f}")

    ground_truth_scan = dd.extract_gt_content(dataset, style=['scan'], region_types=region_types)
    ground_truth_scan = {key : remove_html_tables(item) for key, item in ground_truth_scan.items() if key in prediction}
    mean_cer_scan = dm.mean_cer(ground_truth_scan, prediction)
    mean_wer_scan = dm.mean_wer(ground_truth_scan, prediction)
    print(f"Mean CER scan: {mean_cer_scan:.4f} Mean wER scan: {mean_wer_scan:.4f}")



def teds_benchmarking(dataset, prediction_path, region_types=['table']):
    prediction = dd.load_results(prediction_path)
    # prediction = {key : remove_table_from_text(item) for key, item in prediction.items()}

    ground_truth = dd.extract_gt_content(dataset, region_types=region_types)
    ground_truth = {key : keep_only_html_tables(item) for key, item in ground_truth.items() if key in prediction}

    mean_teds = dm.mean_teds(ground_truth, prediction)
    print(f"Mean TEDS: {mean_teds:.4f}")


def moss_benchmarking(dataset, prediction_path, region_types=['text', 'table', 'chart']):
    prediction = dd.load_results(prediction_path)

    ground_truth = dd.extract_gt_titles(dataset, region_types=region_types)
    ground_truth = {key : item for key, item in ground_truth.items() if key in prediction}

    moss = dm.mean_oss(ground_truth, prediction)
    print(f"Mean OSS: {moss:.4f}")

def class_benchmarking(dataset, prediction_path):
    prediction = dd.load_results(prediction_path)
    ground_truth = {item['id'] : item['category'] for item in dataset if item['id'] in prediction}
    
    acc = dm.accuracy(ground_truth, prediction)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    dataset = dd.download_dataset()['train']

    # print("OCR Benchmarking on text regions")
    # ocr_benchmarking(dataset, 'results/ocr_text.jsonl', region_types=['text'])

    # print("OCR Benchmarking on whole document")
    # ocr_benchmarking(dataset, 'results/ocr_whole_doc.jsonl', region_types=['text'])

    # print("OCR Benchmarking on text regions - Qwen2VL_4bit")
    # ocr_benchmarking(dataset, 'results/ocr_whole_doc_qwen2vl_4bit.jsonl', region_types=['text'])

    # print("OCR Benchmarking on whole document - Qwen2VL_4bit")
    # ocr_benchmarking(dataset, 'results/ocr_whole_doc_qwen2vl_4bit.jsonl', region_types=['text', 'table', 'chart'])

    # print("TEDS Benchmarking on whole document - Qwen2VL_4bit")
    # teds_benchmarking(dataset, 'results/table_whole_doc_qwen2vl_4bit.jsonl', region_types=['table'])

    # print("TEDS Benchmarking on whole document - Qwen2VL")
    # teds_benchmarking(dataset, 'results/table_whole_doc_qwen2vl.jsonl', region_types=['table'])

    # print("Layout Benchmarking on whole document - Qwen2VL_4bit")
    # moss_benchmarking(dataset, 'results/layout_whole_doc_qwen2vl_4bit.jsonl', region_types=['text', 'table', 'chart'])

    # print("Layout Benchmarking on whole document - Qwen2VL")
    # moss_benchmarking(dataset, 'results/layout_whole_doc_qwen2vl.jsonl', region_types=['text', 'table', 'chart'])
    
    print("Class Benchmarking")
    class_benchmarking(dataset, 'results/class_whole_doc_qwen2vl_4bit.jsonl')