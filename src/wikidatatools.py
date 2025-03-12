import requests
import json
import re
import mwparserfromhell

def clean_wiki_text(wiki_text):
    """
    Clean Wikipedia text using mwparserfromhell to remove MediaWiki markup.
    """
    parsed_text = mwparserfromhell.parse(wiki_text)
    cleaned_text = parsed_text.strip_code()

    # Remove auxiliary sections using regex
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

    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def fetch_all_articles(category, language="uk", batch_size=10, max_articles=50, output_file="articles.jsonl"):
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    total_fetched = 0
    cmcontinue = None  # Used for pagination

    with open(output_file, "w", encoding="utf-8") as file:
        while total_fetched < max_articles:
            # Fetch articles from the category
            category_params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Категорія:{category}",
                "cmlimit": batch_size,
                "format": "json"
            }
            if cmcontinue:
                category_params["cmcontinue"] = cmcontinue

            category_response = requests.get(base_url, params=category_params)
            category_response.raise_for_status()
            category_data = category_response.json()

            category_members = category_data.get("query", {}).get("categorymembers", [])
            if not category_members:
                print("No more articles found in the category.")
                break  # Stop if no more articles are found

            for item in category_members:
                if total_fetched >= max_articles:
                    break  # Stop when we reach max_articles

                page_id = item["pageid"]
                title = item["title"]

                print(f"Processing article: {title}")

                # Fetch abstract (summary text)
                abstract_params = {
                    "action": "query",
                    "pageids": page_id,
                    "prop": "extracts",
                    "exintro": True,  # Get only the introduction (abstract)
                    "explaintext": True,
                    "format": "json"
                }
                abstract_response = requests.get(base_url, params=abstract_params)
                abstract_response.raise_for_status()
                abstract_data = abstract_response.json()

                page_abstract = abstract_data["query"]["pages"].get(str(page_id), {}).get("extract", "")

                # Skip articles with no abstract or very short abstract
                if not page_abstract or len(page_abstract) < 50:
                    print(f"Skipping {title}: Abstract too short")
                    continue  # Skip and move to the next article

                # Fetch full content
                content_params = {
                    "action": "query",
                    "pageids": page_id,
                    "prop": "revisions",
                    "rvslots": "*",
                    "rvprop": "content",
                    "format": "json",
                }
                content_response = requests.get(base_url, params=content_params)
                content_response.raise_for_status()
                content_data = content_response.json()

                page = content_data["query"]["pages"].get(str(page_id), {})
                if "revisions" not in page:
                    print(f"Skipping {title}: No full content available")
                    continue

                raw_wikitext = page["revisions"][0]["slots"]["main"]["*"]

                # Clean the extracted MediaWiki text
                cleaned_content = clean_wiki_text(raw_wikitext)

                # Skip articles with very short content
                if len(cleaned_content) < 100:
                    print(f"Skipping {title}: Content too short")
                    continue  # Skip and move to the next article

                # Save the article
                article = {
                    "category": category,
                    "title": title,
                    "abstract": page_abstract,
                    "content": cleaned_content,
                }

                file.write(json.dumps(article, ensure_ascii=False) + "\n")
                total_fetched += 1
                print(f"Saved article {total_fetched}: {title}")

            # Check for pagination
            cmcontinue = category_data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                print("No more pages left to fetch.")
                break  # Stop if no more pages are available

    return total_fetched

if __name__ == "__main__":
    total = fetch_all_articles("Політика", language="uk", batch_size=10, max_articles=50, output_file="politics_articles_filtered.jsonl")
    print(f"Saved {total} articles to physics_articles_filtered.jsonl")
