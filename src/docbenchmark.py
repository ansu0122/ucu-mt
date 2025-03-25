import docdataset as dd
import docmeasure as dm


def ocr_benchmarking(prediction_path, region_types=['text', 'table']):
    prediction = dd.load_results(prediction_path)
    dataset = dd.download_dataset()['train']

    ground_truth = dd.extract_gt_content(dataset, region_types=region_types)
    mean_cer = dm.mean_cer(ground_truth, prediction)
    mean_wer = dm.mean_wer(ground_truth, prediction)
    print(f"Mean CER: {mean_cer:.4f} Mean WER: {mean_wer:.4f}")

    ground_truth_print = dd.extract_gt_content(dataset, style=['print'], region_types=region_types)
    mean_cer_print = dm.mean_cer(ground_truth_print, prediction)
    mean_wer_print = dm.mean_wer(ground_truth_print, prediction)
    print(f"Mean CER print: {mean_cer_print:.4f} Mean WER print: {mean_wer_print:.4f}")

    ground_truth_hand = dd.extract_gt_content(dataset, style=['hand'], region_types=region_types)
    mean_cer_hand = dm.mean_cer(ground_truth_hand, prediction)
    mean_wer_hand = dm.mean_wer(ground_truth_hand, prediction)
    print(f"Mean CER hand: {mean_cer_hand:.4f} Mean WER hand: {mean_wer_hand:.4f}")

    ground_truth_scan = dd.extract_gt_content(dataset, style=['scan'], region_types=region_types)
    mean_cer_scan = dm.mean_cer(ground_truth_scan, prediction)
    mean_wer_scan = dm.mean_wer(ground_truth_scan, prediction)
    print(f"Mean CER scan: {mean_cer_scan:.4f} Mean wER scan: {mean_wer_scan:.4f}")


if __name__ == "__main__":
    print("OCR Benchmarking on text regions")
    ocr_benchmarking('results/ocr_text.jsonl', region_types=['text'])

    print("OCR Benchmarking on whole document")
    ocr_benchmarking('results/ocr_whole_doc.jsonl', region_types=['text', 'table'])
    
