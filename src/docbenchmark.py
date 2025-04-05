import re
import string_util as su
import docdataset as dd
import docmeasure as dm


def eval_ocr(dataset, prediction_path, region_types=['text', 'table']):
    # Load and clean predictions
    prediction = dd.load_results(prediction_path)
    prediction = {key: su.stip_md_table(item) for key, item in prediction.items()}

    # Extract and clean ground truth for all styles
    ground_truth = dd.extract_content(dataset, region_types=region_types)
    ground_truth = {key: su.strip_html_table(item) for key, item in ground_truth.items() if key in prediction}

    mean_cer = dm.mean_cer(ground_truth, prediction)
    mean_wer = dm.mean_wer(ground_truth, prediction)
    print(f"Mean CCR: {(1 - mean_cer) * 100:.4f}% | Mean WCR: {(1 - mean_wer) * 100:.4f}%")

    # Evaluate by style
    for style in ['print', 'hand', 'scan']:
        ground_truth_style = dd.extract_content(dataset, style=[style], region_types=region_types)
        ground_truth_style = {key: su.strip_html_table(item) for key, item in ground_truth_style.items() if key in prediction}
        
        mean_cer_style = dm.mean_cer(ground_truth_style, prediction)
        mean_wer_style = dm.mean_wer(ground_truth_style, prediction)
        
        print(f"Mean CCR ({style}): {(1 - mean_cer_style) * 100:.4f}% | Mean WCR ({style}): {(1 - mean_wer_style) * 100:.4f}%")


def eval_table_extraction(dataset, prediction_path, region_types=['table']):
    # Load predictions
    prediction = dd.load_results(prediction_path)

    # Extract ground truth and format tables
    ground_truth = dd.extract_content(dataset, region_types=region_types)
    ground_truth = {key: su.fetch_html_table(item) for key, item in ground_truth.items() if key in prediction}

    mean_teds = dm.mean_teds(ground_truth, prediction)
    print(f"Mean TEDS (overall): {mean_teds * 100:.4f}%")

    # Evaluate TEDS by style
    for style in ['print', 'hand', 'scan']:
        ground_truth_style = dd.extract_content(dataset, style=[style], region_types=region_types)
        ground_truth_style = {
            key: su.fetch_html_table(item) 
            for key, item in ground_truth_style.items() if key in prediction
        }

        mean_teds_style = dm.mean_teds(ground_truth_style, prediction)
        print(f"Mean TEDS ({style}): {mean_teds_style * 100:.4f}%")


def eval_layout_analysis(dataset, prediction_path, region_types=['text', 'table', 'chart']):
    # Load predictions
    prediction = dd.load_results(prediction_path)

    # Extract ground truth titles
    ground_truth = dd.extract_titles(dataset, region_types=region_types)
    ground_truth = {key: item for key, item in ground_truth.items() if key in prediction}

    moss = dm.mean_oss(ground_truth, prediction)
    print(f"Mean OSS (overall): {moss * 100:.4f}%")

    # Evaluate OSS by style
    for style in ['print', 'hand', 'scan']:
        ground_truth_style = dd.extract_titles(dataset, style=[style], region_types=region_types)
        ground_truth_style = {
            key: item for key, item in ground_truth_style.items() if key in prediction
        }

        moss_style = dm.mean_oss(ground_truth_style, prediction)
        print(f"Mean OSS ({style}): {moss_style * 100:.4f}%")


def eval_classification(dataset, prediction_path):
    # Load predictions
    prediction = dd.load_results(prediction_path)

    # Extract overall ground truth
    ground_truth = {item['id']: item['category'] for item in dataset if item['id'] in prediction}
    acc = dm.accuracy(ground_truth, prediction)
    print(f"Accuracy (overall): {acc * 100:.4f}%")

    # Optional: Style-wise accuracy
    for style in ['print', 'hand', 'scan']:
        ground_truth_style = {
            item['id']: item['category']
            for item in dataset
            if item.get('style') == style and item['id'] in prediction
        }

        if ground_truth_style:
            acc_style = dm.accuracy(ground_truth_style, prediction)
            print(f"Accuracy ({style}): {acc_style * 100:.4f}%")

def ocr_bench():
    # Tesseract
    print("OCR - Text - Tesseract")
    eval_ocr(dataset, 'results/ocr_text.jsonl', region_types=['text'])

    print("OCR - Document - Tesseract")
    eval_ocr(dataset, 'results/ocr_whole_doc.jsonl')

    # Qwen2VL-4bit
    print("OCR - Text - Qwen2VL_4bit")
    eval_ocr(dataset, 'results/ocr_text_qwen2vl_4bit.jsonl', region_types=['text'])

    print("OCR - Text V2 - Qwen2VL_4bit")
    eval_ocr(dataset, 'results/ocr_text_qwen2vl_4bit_v2.jsonl', region_types=['text'])

    print("OCR - Document - Qwen2VL_4bit")
    eval_ocr(dataset, 'results/ocr_whole_doc_qwen2vl_4bit.jsonl')

    print("OCR - Document V2 - Qwen2VL_4bit")
    eval_ocr(dataset, 'results/ocr_whole_doc_qwen2vl_4bit_v2.jsonl')

    # Qwen2VL
    print("OCR - Text V2 - Qwen2VL")
    eval_ocr(dataset, 'results/ocr_text_qwen2vl.jsonl', region_types=['text'])

    print("OCR - Document V2 - Qwen2VL")
    eval_ocr(dataset, 'results/ocr_whole_doc_qwen2vl.jsonl')

    # Qwen2.5VL
    print("OCR - Text V2 - Qwen2.5VL")
    eval_ocr(dataset, 'results/ocr_text_qwen25vl.jsonl', region_types=['text'])

    print("OCR - Document V2 - Qwen2.5VL")
    eval_ocr(dataset, 'results/ocr_whole_doc_qwen25vl.jsonl')

    # Phi4VL
    print("OCR - Text V2 - Phi4VL")
    eval_ocr(dataset, 'results/ocr_text_phi4vl.jsonl', region_types=['text'])

    print("OCR - Document V2 - Phi4VL")
    eval_ocr(dataset, 'results/ocr_whole_doc_phi4vl.jsonl')

    # MistralOCR
    print("OCR - Text V2 - MistralOCR")
    eval_ocr(dataset, 'results/ocr_text_mistralocr.jsonl', region_types=['text'])

    print("OCR - Document V2 - MistralOCR")
    eval_ocr(dataset, 'results/ocr_whole_doc_mistralocr.jsonl')

    # Gemeni2.0 Flash


def tabext_bench():
    # Qwen2VL-4bit
    print("Table Extraction - Table - Qwen2VL_4bit")
    eval_table_extraction(dataset, 'results/table_table_qwen2vl_4bit.jsonl', region_types=['table'])

    print("Table Extraction - Document - Qwen2VL_4bit")
    eval_table_extraction(dataset, 'results/table_whole_doc_qwen2vl_4bit.jsonl')

    # Qwen2VL
    print("Table Extraction - Table - Qwen2VL")
    eval_table_extraction(dataset, 'results/table_table_qwen2vl.jsonl', region_types=['table'])

    print("Table Extraction - Document - Qwen2VL")
    eval_table_extraction(dataset, 'results/table_whole_doc_qwen2vl.jsonl')

    # Qwen2.5VL
    print("Table Extraction - Table - Qwen2.5VL")
    eval_table_extraction(dataset, 'results/table_table_qwen25vl.jsonl', region_types=['table'])

    print("Table Extraction - Document - Qwen2.5VL")
    eval_table_extraction(dataset, 'results/table_whole_doc_qwen25vl.jsonl')

    # Phi4VL
    print("Table Extraction - Table - Phi4VL")
    eval_table_extraction(dataset, 'results/table_table_phi4vl.jsonl', region_types=['table'])

    print("Table Extraction - Document - Phi4VL")
    eval_table_extraction(dataset, 'results/table_whole_doc_phi4vl.jsonl')

    # MistralOCR
    print("Table Extraction - Table - MistralOCR")
    eval_table_extraction(dataset, 'results/table_table_mistralocr.jsonl', region_types=['table'])

    print("Table Extraction - Document - MistralOCR")
    eval_table_extraction(dataset, 'results/table_whole_doc_mistralocr.jsonl')

    # Gemeni2.0 Flash


def layout_bench():
    # Qwen2VL-4bit
    print("Layout Analysis - Document - Qwen2VL_4bit")
    eval_layout_analysis(dataset, 'results/layout_whole_doc_qwen2vl_4bit.jsonl')

    # Qwen2VL
    print("Layout Analysis - Document - Qwen2VL")
    eval_layout_analysis(dataset, 'results/layout_whole_doc_qwen2vl.jsonl')

    # Qwen2.5VL
    print("Layout Analysis - Document - Qwen2.5VL")
    eval_layout_analysis(dataset, 'results/layout_whole_doc_qwen25vl.jsonl')

    # Phi4VL
    print("Layout Analysis - Document - Phi4VL")
    eval_layout_analysis(dataset, 'results/layout_whole_doc_phi4vl.jsonl')

    # MistralOCR
    print("Layout Analysis - Document - MistralOCR")
    eval_layout_analysis(dataset, 'results/layout_whole_doc_mistralocr.jsonl')

    # Gemeni2.0 Flash


def class_bench():
    # Qwen2VL-4bit
    print("Classification - Document - Qwen2VL_4bit")
    eval_classification(dataset, 'results/class_whole_doc_qwen2vl_4bit.jsonl')

    # Qwen2VL
    print("Classification - Document - Qwen2VL")
    eval_classification(dataset, 'results/class_whole_doc_qwen2vl.jsonl')

    # Qwen2.5VL
    print("Classification - Document - Qwen2.5VL")
    eval_classification(dataset, 'results/class_whole_doc_qwen25vl.jsonl')

    # Phi4VL
    print("Classification - Document - Phi4VL")
    eval_classification(dataset, 'results/class_whole_doc_phi4vl.jsonl')

    # MistralOCR
    print("Classification - Document - MIstralOCR")
    eval_classification(dataset, 'results/class_whole_doc_mistralocr.jsonl')

    # Gemeni2.0 Flash


if __name__ == "__main__":
    dataset = dd.download_dataset()['train']

    

    

    
    
   