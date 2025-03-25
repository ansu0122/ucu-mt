import re
import numpy as np
from Levenshtein import distance
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

class TreeNode:
    """Represents a node in a tree."""
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children else []

def tree_edit_distance(T1, T2):
    """
    Compute the Tree Edit Distance (TED) between two trees using a custom implementation.
    :param T1: Root node of first tree
    :param T2: Root node of second tree
    :return: The tree edit distance (TED) between T1 and T2
    """
    if T1 is None:
        return tree_size(T2)
    if T2 is None:
        return tree_size(T1)
    
    m, n = len(T1.children), len(T2.children)
    dp = np.zeros((m + 1, n + 1))
    
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + tree_size(T1.children[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + tree_size(T2.children[j - 1])
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if T1.children[i - 1].label == T2.children[j - 1].label else 1
            dp[i][j] = min(
                dp[i - 1][j] + tree_size(T1.children[i - 1]),
                dp[i][j - 1] + tree_size(T2.children[j - 1]),
                dp[i - 1][j - 1] + tree_edit_distance(T1.children[i - 1], T2.children[j - 1]) + cost
            )
    
    return dp[m][n]

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
    Converts an HTML string into a custom tree structure, stripping all attributes.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    def traverse(node):
        """
        Recursively traverse the HTML tree and convert it into a custom TreeNode structure,
        ignoring attributes.
        """
        if node.name:
            children = [traverse(child) for child in node.children if child.name]
            return TreeNode(node.name, children)
        return None
    
    return traverse(soup)


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
    html1 = """<table class='table'><tr><td>Cell1</td><td>Cell2</td></tr><tr><td>Cell3</td><td>Cell4</td></tr></table>"""
    html2 = """<table><tr><td>Cell1</td><td>Cell2</td></tr><tr><td>Cell3</td><td>Cell4</td></tr></table>"""
    tree1 = html_to_tree(html1)
    tree2 = html_to_tree(html2)

    ted_distance = tree_edit_distance(tree1, tree2)
    print("Tree Edit Distance (TED):", ted_distance)

    predicted_titles = ["Introduction", "Method", "Results and Discussion", "Conclusion"]
    ground_truth_titles = ["Introduction", "Methodology", "Results and Discussion", "Conclusion"]

    sequence_similarity = ordered_sequence_similarity(predicted_titles, ground_truth_titles)

    print("Cosine Similarity (Sequence-Level):", sequence_similarity)
