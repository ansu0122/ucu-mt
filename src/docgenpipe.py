import os
import json
import argparse
import random
from markdown2 import markdown
from playwright.sync_api import sync_playwright
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# --- Step 1: Read JSONL Records ---
def read_jsonl(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- Step 2: Read Markdown Tables ---
def read_markdown_tables(table_paths):
    tables = []
    for path in table_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                tables.append(f.read())
        except FileNotFoundError:
            print(f"Warning: Table file {path} not found.")
    return tables

# --- Step 3: Simplify Tables with LLM ---
class SimplifiedTable(BaseModel):
    html_table: str = Field(description="Simplified HTML table version of the original.")

def simplify_table(llm, table_data):
    parser = PydanticOutputParser(pydantic_object=SimplifiedTable)

    prompt_template = """
    Маючи наступну таблицю в форматі HTML:
    {table_data}

    Спрости її зберігаючи наявну інформацію. Позбудься непортібних тегів і атрибутів.
    Переконайся, що таблиця читабельна.
    Відформатуй результат як JSON використовуючи наступну схему:\n{format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["table_data"],
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    return chain.invoke({"table_data": table_data}).html_table

# --- Step 4: Generate Markdown Summary ---

class ArticleSections(BaseModel):
    section_1_title: str = Field(description="Заголовок для огляду теми")
    section_1_content: str = Field(description="Огляд заданої теми в деталях")

    section_2_title: str = Field(description="Заголовок для ключової інформації")
    section_2_content: str = Field(description="Ключова інформація або приклади в деталях")

    section_3_title: str = Field(description="Заголовок для релевантних даних")
    section_3_content: str = Field(description="Релевантні дані або додаткові деталі")

    table_title: str = Field(description="Заголовок для таблиці")
    table_content: str = Field(description="Розгорнута таблиця на основі ключових даних в форматі HTML")

def generate_content(llm, title, adstract, content) -> ArticleSections:
    """
    Uses LLM to generate structured content sections with titles based on:
    - Article Title
    - Full Article Text
    - Wikipedia URL for external reference
    """
    parser = PydanticOutputParser(pydantic_object=ArticleSections)

    prompt_template = """
    На основі наступної статті:

    **Заголовок:** {title}
    
    **Вступ:** {adstract}
    
    **Контент:** {content}

    Згенеруй структуровані розділи та таблицю з чіткими **заголовками** та **вмістом**.
    - **Заголовки** мають бути **чіткі та описові**.
    - **Контент** має бути **структурований, базуватись на фактах і доречний**.
    - **Не додавай непотрібних представлень** — переходь відразу до ключових пунктів.

    Відповідь повертай у форматі JSON використовуючи цю схему:\n{format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["title", "article_text", "wiki_url"],
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    content = prompt | llm | parser

    return content.invoke({
        "title": title,
        "adstract": adstract,
        "content": content
    })

def load_templates(template_dir="dataset/templates"):
    """
    Loads all Markdown templates from the specified directory.
    Returns a dictionary where keys are filenames (without .md) and values are the content.
    """
    templates = {}

    for filename in os.listdir(template_dir):
        if filename.endswith(".md"):
            template_name = os.path.splitext(filename)[0]  # Remove .md extension
            with open(os.path.join(template_dir, filename), "r", encoding="utf-8") as f:
                templates[template_name] = f.read()

    return templates

class MarkdownSummary(BaseModel):
    markdown_content: str = Field(description="Markdown контент зі структурованими розділами")

def generate_summary(title: str, abstract: str, content: dict):
    TEMPLATES = load_templates()

    selected_template_name = random.choice(list(TEMPLATES.keys()))
    selected_template = TEMPLATES[selected_template_name]

    print(f"Using template: {selected_template_name}")

    content.update({"title": title, "abstract": abstract})

    for key, value in content.items():
        selected_template = selected_template.replace(f"{{{key}}}", str(value))

    return selected_template


# --- Step 5: Save Markdown as PDF & PNG ---
def save_as_pdf(markdown_text, output_dir, filename):
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    raw_html = markdown(markdown_text)
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                width: 100%;
                margin: 0;
                padding: 0;
            }}
            .wrapper {{
                max-width: 900px;
                margin: auto;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="wrapper">
            {raw_html}
        </div>
    </body>
    </html>
    """

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content)
        page.pdf(path=pdf_path, format="A4", landscape=False, print_background=True)
        browser.close()

    return pdf_path

def save_as_png(markdown_text, output_dir, filename):
    png_path = os.path.join(output_dir, f"{filename}.png")
    raw_html = markdown(markdown_text)

    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                width: 100%;
                margin: 0;
                padding: 0;
            }}
            .wrapper {{
                max-width: 900px;
                margin: auto;
                padding: 10px;
            }}
            table, th, td {{
                border: 1px solid black;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <div class="wrapper">
            {raw_html}
        </div>
    </body>
    </html>
    """

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content)
        page.screenshot(path=png_path, full_page=True)
        browser.close()

    return png_path




# --- Step 6: Process JSONL and Save Results ---
def process_jsonl(input_file, output_file, llm_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    llm = ChatOpenAI(model=llm_model)

    records = read_jsonl(input_file)
    processed_records = []

    for i, record in enumerate(records[:1]):
        print(f"Processing record {i+1}/{len(records)}: {record.get('title', 'Unknown Title')}")

        title = record.get("title", "")
        abstract = record.get("abstract", "")
        content = record.get("content", abstract)

        # Generate structured content sections using LLM
        gen_content = generate_content(llm, title, abstract, content)

        markdown_content = generate_summary(title, abstract, gen_content.model_dump())

        filename = f"{title.replace(' ', '_')}_{i+1}"
        png_path = save_as_png(markdown_content, output_dir, filename)

        record["summary_markdown"] = markdown_content
        record["png_path"] = png_path
        processed_records.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Processed {len(records)} records. Results saved to {output_file}")


# --- Main Execution ---
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process Wikipedia JSONL records and generate summaries.")
    # parser.add_argument('--input_file', type=str, required=True, help="Path to input JSONL file")
    # parser.add_argument('--output_file', type=str, required=True, help="Path to save processed JSONL file")
    # parser.add_argument('--llm_model', type=str, required=True, help="LLM model name")
    # parser.add_argument('--output_dir', type=str, required=True, help="Directory to save PDFs and PNGs")

    # args = parser.parse_args()

    # process_jsonl(
    #     input_file=args.input_file,
    #     output_file=args.output_file,
    #     llm_model=args.llm_model,
    #     output_dir=args.output_dir
    # )

    process_jsonl(
        input_file="dataset/economy_articles_filtered.jsonl",
        output_file="economy_processed.jsonl",
        llm_model="gpt-4o",
        output_dir="dataset"
    )

    # python src/datagenpipe.py \
    #     --input_file "dataset/economy_articles_filtered.jsonl" \
    #     --output_file "economy_processed.jsonl" \
    #     --llm_model "gpt-4o" \
    #     --output_dir "dataset"
