import re

def stip_md_table(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Skip lines that look like table rows (contain at least 2 '|' characters)
        if line.count('|') >= 2:
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def strip_html_table(text: str) -> str:
    # Remove <table>...</table> blocks (non-greedy match)
    return re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


def fetch_html_table(text: str) -> str:
    # Find all <table>...</table> blocks
    tables = re.findall(r'<table.*?>.*?</table>', text, flags=re.DOTALL | re.IGNORECASE)
    return '\n\n'.join(tables).strip()