import requests
import json
import re
import mwparserfromhell

def clean_wiki_text(wiki_text):
    parsed_text = mwparserfromhell.parse(wiki_text)
    cleaned_text = parsed_text.strip_code()

    # Remove auxiliary sections
    auxiliary_sections = [
        r"== Див\. також ==.*",
        r"== Примітки ==.*",
        r"== Джерела ==.*",
        r"== Література ==.*",
        r"== Посилання ==.*",
        r"== Зовнішні посилання ==.*"
    ]
    for section in auxiliary_sections:
        cleaned_text = re.sub(section, "", cleaned_text, flags=re.DOTALL)

    return re.sub(r'\s+', ' ', cleaned_text).strip()

def fetch_all_articles(category, language="uk", batch_size=10, max_articles=100, output_file="articles.jsonl"):
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    seen_pages = set()
    seen_categories = set()
    total_fetched = 0
    category_initial = category

    def fetch_category(category_title):
        nonlocal total_fetched
        if category_title in seen_categories or total_fetched >= max_articles:
            return
        seen_categories.add(category_title)

        print(f"\nExploring category: {category_title}")
        cmcontinue = None
        while total_fetched < max_articles:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Категорія:{category_title}",
                "cmlimit": batch_size,
                "format": "json",
                "cmcontinue": cmcontinue if cmcontinue else None,
            }

            response = requests.get(base_url, params={k: v for k, v in params.items() if v is not None})
            response.raise_for_status()
            data = response.json()

            members = data.get("query", {}).get("categorymembers", [])
            if not members:
                break

            for member in members:
                if total_fetched >= max_articles:
                    break

                page_id = member["pageid"]
                title = member["title"]

                if title.startswith("Категорія:"):
                    # Recurse into subcategory
                    subcat_name = title.replace("Категорія:", "")
                    fetch_category(subcat_name)
                elif page_id not in seen_pages:
                    seen_pages.add(page_id)
                    print(f"Processing article: {title}")
                    try:
                        # Fetch abstract
                        abstract_params = {
                            "action": "query",
                            "pageids": page_id,
                            "prop": "extracts",
                            "exintro": True,
                            "explaintext": True,
                            "format": "json"
                        }
                        abstract_resp = requests.get(base_url, params=abstract_params)
                        abstract_resp.raise_for_status()
                        abstract_data = abstract_resp.json()
                        abstract = abstract_data["query"]["pages"].get(str(page_id), {}).get("extract", "")

                        if not abstract or len(abstract) < 100:
                            print(f"Skipping {title}: Abstract too short")
                            continue
                        if len(abstract) > 2000:
                            print(f"Skipping {title}: Abstract too long ({len(abstract)} characters)")
                            continue

                        # Fetch full wikitext
                        content_params = {
                            "action": "query",
                            "pageids": page_id,
                            "prop": "revisions",
                            "rvslots": "*",
                            "rvprop": "content",
                            "format": "json"
                        }
                        content_resp = requests.get(base_url, params=content_params)
                        content_resp.raise_for_status()
                        content_data = content_resp.json()

                        page = content_data["query"]["pages"].get(str(page_id), {})
                        if "revisions" not in page:
                            print(f"Skipping {title}: No content found")
                            continue

                        raw_text = page["revisions"][0]["slots"]["main"]["*"]
                        cleaned_content = clean_wiki_text(raw_text)

                        if len(cleaned_content) < 1000:
                            print(f"Skipping {title}: Content too short")
                            continue

                        if len(cleaned_content.split()) > 5000:
                            print(f"Skipping {title}: Content too long ({len(cleaned_content.split())} words)")
                            continue

                        # Save article
                        article = {
                            "category": category_initial,
                            "title": title,
                            "abstract": abstract,
                            "content": cleaned_content,
                        }
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(article, ensure_ascii=False) + "\n")

                        total_fetched += 1
                        print(f"Saved {total_fetched}: {title}")
                    except Exception as e:
                        print(f"Error fetching {title}: {e}")

            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break

    fetch_category(category)
    return total_fetched

if __name__ == "__main__":
    # output_file="data/economy.jsonl"
    # total = fetch_all_articles("Економіка", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/politics.jsonl"
    # total = fetch_all_articles("Політика", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/literature.jsonl"
    # total = fetch_all_articles("Література", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/science.jsonl"
    # total = fetch_all_articles("Наука", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/art.jsonl"
    # total = fetch_all_articles("Мистецтво", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/education.jsonl"
    # total = fetch_all_articles("Освіта", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/healthcare.jsonl"
    # total = fetch_all_articles("Медицина", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/finance.jsonl"
    # total = fetch_all_articles("Фінанси", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/coding.jsonl"
    # total = fetch_all_articles("Програмування", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    # output_file="data/math.jsonl"
    # total = fetch_all_articles("Математика", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    # print(f"Saved {total} articles to {output_file}")

    output_file="data/west_art.jsonl"
    total = fetch_all_articles("Західне мистецтво", language="uk", batch_size=10, max_articles=100, output_file=output_file)
    print(f"Saved {total} articles to {output_file}")

