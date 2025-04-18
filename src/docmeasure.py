import re
import numpy as np
from Levenshtein import distance
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zss import Node, simple_distance
from sentence_transformers import SentenceTransformer

vectorizer = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_cer(gt_text, ocr_text):
    edit_distance = distance(gt_text, ocr_text)
    cer = edit_distance / max(1, len(gt_text))
    return cer


def mean_cer(ground_truth: dict, predictions: dict, verbose: bool = False) -> float:
    matches = _match_gt_results(ground_truth, predictions)

    total_cer = 0.0

    for match in matches:
        cer = calculate_cer(match['ground_truth'], match['prediction'])
        if verbose:
            print(f"ID: {match['id']}, CER: {cer:.4f}")
            
        total_cer += cer

    return total_cer / len(matches) if matches else 0.0


def _normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

def _levenshtein(seq1, seq2):
    """Compute Levenshtein distance between two sequences (word-level)."""
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # deletion
                    dp[i][j - 1],    # insertion
                    dp[i - 1][j - 1] # substitution
                )
    return dp[n][m]

def calculate_wer(gt_text, ocr_text):
    gt_words = _normalize_text(gt_text).split()
    ocr_words = _normalize_text(ocr_text).split()

    edit_distance = _levenshtein(gt_words, ocr_words)
    wer = edit_distance / max(1, len(gt_words))
    return wer


def mean_wer(ground_truth: dict, predictions: dict, verbose: bool = False) -> float:
    matches = _match_gt_results(ground_truth, predictions)

    total_wer = 0.0

    for match in matches:
        wer = calculate_wer(match['ground_truth'], match['prediction'])
        if verbose:
            print(f"ID: {match['id']}, WER: {wer:.4f}")
        
        total_wer += wer

    return total_wer / len(matches) if matches else 0.0


def _match_gt_results(ground_truth: dict, prediction: dict):
    comparisons = []
    for doc_id, gt in ground_truth.items():
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


def mean_teds(ground_truth: dict, predictions: dict, verbose: bool = False) -> float:
    matches = _match_gt_results(ground_truth, predictions)

    total_teds = 0.0

    for match in matches:
        score = teds(match['ground_truth'], match['prediction'])
        if verbose:
            print(f"ID: {match['id']}, TEDS: {score:.4f}")
        total_teds += score

    return total_teds / len(matches) if matches else 0.0


def mean_oss(ground_truth: dict, predictions: dict, verbose: bool = False) -> float:
    matches = _match_gt_results(ground_truth, predictions)

    total_oss = 0.0

    for match in matches:
        score = ordered_sequence_similarity(match['ground_truth'], match['prediction'])
        if verbose:
            print(f"ID: {match['id']}, OSS: {score:.4f}")
            
        total_oss += score

    return total_oss / len(matches) if matches else 0.0


def ordered_sequence_similarity(
    ground_truth: list, 
    predictions: list, 
    threshold: float = 0.75, 
    max_roll: int = 5,
    verbose: bool = False
) -> float:
    if not predictions and not ground_truth:
        return 1.0
    elif not predictions or not ground_truth:
        return 0.0

    pred_vec = vectorizer.encode(predictions)
    gt_vec = vectorizer.encode(ground_truth)

    total_score = 0.0
    matches = 0

    for i in range(len(gt_vec)):
        gt_title_vec = gt_vec[i].reshape(1, -1)
        score = 0.0

        for offset in range(max_roll + 1):
            pred_index = i + offset
            if pred_index >= len(pred_vec):
                break

            pred_title_vec = pred_vec[pred_index].reshape(1, -1)
            sim = cosine_similarity(gt_title_vec, pred_title_vec)[0][0]

            if sim >= threshold or offset == 0:
                # Apply soft penalty for distance from ideal position
                decay = 1.0 / (1 + offset)
                score = sim * decay
                if verbose:
                    print(f"GT[{i}] matched to Pred[{pred_index}] with sim={sim:.2f}, decay={decay:.2f}")
                break  # take the first match above threshold (or fallback to offset=0 match)

        total_score += score
        matches += 1

    return total_score / matches if matches > 0 else 0.0


def accuracy(ground_truth: dict, predictions: dict) -> float:
    matches = _match_gt_results(ground_truth, predictions)
    correct = sum(1 for match in matches if match['ground_truth'] == match['prediction'])

    return correct / len(ground_truth) if ground_truth else 0.0

if __name__ == "__main__":

    predicted_titles = ["Introduction", "Conclusion", "Results and Discussion", "Methodology"]
    ground_truth_titles = ["Introduction", "Methodology", "Results and Discussion", "Conclusion"]

    similarity = ordered_sequence_similarity(predicted_titles, ground_truth_titles)

    print("Cosine Similarity (Sequence-Level):", similarity)


    html1 = "<table><tbody><tr><td>Республіка</td><td>Назва фронту</td><td>Дата заснування</td><td>Тип</td></tr><tr><td>РРФСР</td><td>Демократична Росія</td><td>1990</td><td>Національний рух</td></tr><tr><td>УРСР</td><td>Народний Рух України</td><td>1988</td><td>Національний рух</td></tr><tr><td>БРСР</td><td>Відродження</td><td>1989</td><td>Національний рух</td></tr><tr><td>Узбецька РСР</td><td>Єдність</td><td>1988</td><td>Національний рух</td></tr><tr><td>КазРСР</td><td>Народний рух Казахстану</td><td>1989</td><td>Національний рух</td></tr><tr><td>ГРСР</td><td>Комітет національної свободи</td><td>1989</td><td>Національний рух</td></tr><tr><td>АзРСР</td><td>Народний фронт Азербайджану</td><td>1988</td><td>Національний рух</td></tr></tbody></table>"
    html2 = "<table>\n  <thead>\n    <tr>\n      <th>Республіканська партія</th>\n      <th>Назва фронту</th>\n      <th>Дата заснування</th>\n      <th>Тип</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>РРФСР</td>\n      <td>Демократична Росія</td>\n      <td>1990</td>\n      <td>Національний рух</td>\n    </tr>\n    <tr>\n      <td>УРСР</td>\n      <td>Народний Рух України</td>\n      <td>1988</td>\n      <td>Національний рух</td>\n    </tr>\n    <tr>\n      <td>БРСР</td>\n      <td>Відродження</td>\n      <td>1989</td>\n      <td>Національний рух</td>\n    </tr>\n    <tr>\n      <td>Узбецька РСР</td>\n      <td>Єдність</td>\n      <td>1988</td>\n      <td>Національний рух</td>\n    </tr>\n    <tr>\n      <td>КазРСР</td>\n      <td>Народний рух Казахстану</td>\n      <td>1989</td>\n      <td>Національний рух</td>\n    </tr>\n    <tr>\n      <td>ГРСР</td>\n      <td>Комітет національної свободи</td>\n      <td>1989</td>\n      <td>Національний рух</td>\n    </tr>\n    <tr>\n      <td>АзРСР</td>\n      <td>Народний фронт Азербайджану</td>\n      <td>1988</td>\n      <td>Національний рух</td>\n    </tr>\n  </tbody>\n</table>"

    teds = teds(html1, html2)

    print('real TEDS: ', teds)

    tex1 = 'Борiтеся – поборете, Вам Бог помагає!'
    tex2 = 'борiтесь – поборете, Вам Бог помага!'
    print('CER', calculate_cer(tex1, tex2))
    print('WER', calculate_wer(tex1, tex2))


