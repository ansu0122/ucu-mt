import re
import numpy as np
from Levenshtein import distance
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zss import Node, simple_distance


def calculate_cer(gt_text, ocr_text):
    edit_distance = distance(gt_text, ocr_text)
    cer = edit_distance / max(1, len(gt_text))
    return cer


def mean_cer(ground_truth: list, predictions: dict) -> float:
    
    matches = _match_gt_results(ground_truth, predictions)

    total_cer = sum(
        calculate_cer(match['ground_truth'], match['prediction'])
        for match in matches
    )
    return total_cer / len(matches) if matches else 0.0


def _normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


def calculate_wer(gt_text, ocr_text):
    gt_words = _normalize_text(gt_text).split()
    ocr_words = _normalize_text(ocr_text).split()

    edit_distance = distance(" ".join(gt_words), " ".join(ocr_words))
    wer = edit_distance / max(1, len(gt_words))
    return wer


def mean_wer(ground_truth: list, predictions: dict) -> float:
    matches = _match_gt_results(ground_truth, predictions)

    total_wer = sum(
        calculate_wer(match['ground_truth'], match['prediction'])
        for match in matches
    )
    return total_wer / len(matches) if matches else 0.0


def _match_gt_results(ground_truth: list, prediction: dict):
    comparisons = []
    for record in ground_truth:
        doc_id = record["id"]
        gt = record["content"]
        ocr = prediction.get(doc_id, "")
        comparisons.append({
            "id": doc_id,
            "ground_truth": gt,
            "prediction": ocr
        })

    return comparisons


def tree_size(T):
    """
    Compute the number of nodes in a tree.
    :param T: Root node of tree
    :return: Size of the tree (number of nodes)
    """
    if T is None:
        return 0
    return 1 + sum(tree_size(child) for child in T.children)

def html_to_tree(html):
    """
    Converts an HTML string into a zss-compatible tree structure, stripping all attributes.
    """
    soup = BeautifulSoup(html, 'html.parser')

    def traverse(bs_node):
        if bs_node.name is None:
            return None  # Skip non-tag nodes (e.g., strings or comments)

        zss_node = Node(bs_node.name)
        for child in bs_node.children:
            child_node = traverse(child)
            if child_node:
                zss_node.addkid(child_node)
        return zss_node

    # Typically you want to start at <html> or <table>, so pick the main tag
    root_tag = next((c for c in soup.contents if c.name), None)
    return traverse(root_tag) if root_tag else None

def teds(html1, html2):
    t1 = html_to_tree(html1)
    t2 = html_to_tree(html2)

    if t1 is None or t2 is None:
        ted = float('inf')  # or 0, depending on your policy
    else:
        ted = simple_distance(t1, t2)

    return 1 - ted / max(tree_size(t1), tree_size(t2))


def ordered_sequence_similarity(predicted_titles, ground_truth_titles):
    """
    Compute cosine similarity between predicted and ground truth title sequences 
    while preserving order using n-grams in TF-IDF.
    """
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))

    predicted_sequence = " -> ".join(predicted_titles)
    ground_truth_sequence = " -> ".join(ground_truth_titles)

    tfidf_matrix = vectorizer.fit_transform([predicted_sequence, ground_truth_sequence])

    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    return similarity_score


if __name__ == "__main__":

    predicted_titles = ["Introduction", "Method", "Results and Discussion", "Conclusion"]
    ground_truth_titles = ["Introduction", "Methodology", "Results and Discussion", "Conclusion"]

    sequence_similarity = ordered_sequence_similarity(predicted_titles, ground_truth_titles)

    print("Cosine Similarity (Sequence-Level):", sequence_similarity)


    html1 = """<table><tbody><tr><td rowspan="2">Інвестиційні Проекти</td><td colspan="3">Гранична Ефективність Капіталу (%)</td><td colspan="3">Інвестиційний Попит при Різних Процентних Ставках</td></tr><tr><td>20%</td><td>18%</td><td>12%</td><td>8%</td><td>15%</td><td>10%</td></tr><tr><td rowspan="2">Процентна Ставка</td><td>20%</td><td>18%</td><td>12%</td><td>8%</td><td>0%</td><td>Проекти 1 і 2</td></tr><tr><td>Зниження</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>Проекти 1, 2 і 3</td></tr><tr><td rowspan="2">Інвестиційний Попит</td><td>0</td><td>0</td><td>Проекти 1 та 2</td><td>Проекти 1, 2 і 3</td><td>Всі Проекти</td><td>Максимальний</td></tr><tr><td>Зниження</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>Зростання</td></tr></tbody></table>"""
    html2 = """<table>\n  <thead>\n    <tr>\n      <th>Інвестиційні Проекти</th>\n      <th>Границя Ефективність Капіталу (%)</th>\n      <th>Інвестиційний Попит при Різних Процентних Ставках</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>20%</td>\n      <td>18 %</td>\n      <td>12%</td>\n      <td>8%</td>\n      <td>15%</td>\n      <td>10%</td>\n    </tr>\n    <tr>\n      <td>Процентна Ставка</td>\n      <td>20%</td>\n      <td>18 %</td>\n      <td>12%</td>\n      <td>8%</td>\n      <td>0%</td>\n      <td>Проекти 1 і 2</td>\n    </tr>\n    <tr>\n      <td>Зниження</td>\n      <td>N/A</td>\n      <td>N/A</td>\n      <td>N/A</td>\n      <td>N/A</td>\n      <td>Проекти 1, 2 і 3</td>\n    </tr>\n    <tr>\n      <td>Інвестиційний Попит</td>\n      <td>0</td>\n      <td>0</td>\n      <td>Проекти 1 та 2</td>\n      <td>Проекти 1, 2 і 3</td>\n      <td>Всі Проекти</td>\n      <td>Максимальний</td>\n    </tr>\n    <tr>\n      <td>Зниження</td>\n      <td>N/A</td>\n      <td>N/A</td>\n      <td>N/A</td>\n      <td>N/A</td>\n      <td>Зростання</td>\n    </tr>\n  </tbody>\n</table>"""

    teds = teds(html1, html2)

    print('real TEDS: ', teds)
