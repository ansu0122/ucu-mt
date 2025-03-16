import os
import re
import json
import argparse
import random
from markdown2 import markdown
from playwright.sync_api import sync_playwright, Page
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import uuid
import docvision as dv
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

load_dotenv()


def read_jsonl(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def clean_html(html):
    html = re.sub(r'class="[^"]*"', '', html)
    html = re.sub(r'\n\s*', ' ', html)
    return html.strip()

class ArticleSections(BaseModel):
    section_1_title: str = Field(description="Заголовок для огляду теми")
    section_1_content: str = Field(description="Огляд заданої теми в деталях")

    section_2_title: str = Field(description="Заголовок для ключової інформації")
    section_2_content: str = Field(description="Ключова інформація або приклади в деталях")

    section_3_title: str = Field(description="Заголовок для релевантних даних")
    section_3_content: str = Field(description="Релевантні дані або додаткові деталі")

    table_title: str = Field(description="Заголовок для таблиці")
    table_content: str = Field(description=f"Розгорнута таблиця з {random.choice([2, 3, 4])} стовпцями і принаймні {random.choice([2, 3, 4, 5, 6])} рядками на основі ключових даних в форматі HTML")

    svg_chart_title: str = Field(description="Заголовок для графiку")
    svg_chart: str = Field(description="Графiк у форматі SVG, що відображає ключову інформацію")

def generate_content(llm, title, content) -> ArticleSections:
    """
    Uses LLM to generate structured content sections with titles based on:
    - Article Title
    - Full Article Text
    """
    parser = PydanticOutputParser(pydantic_object=ArticleSections)

    prompt_template = """
    На основі контенту наступної статті:

    **Заголовок:** {title}
    
    **Контент:** {content}

    Згенеруй **Українською мовою** структуровані розділи, таблицю та візуалізацію з чіткими **заголовками** та **вмістом**.
    - **Заголовки** мають бути **чіткі та описові**.
    - **Контент** має бути **структурований, базуватись на фактах і доречний**.
    - **Не додавай непотрібних представлень** — переходь відразу до ключових пунктів.

    Відповідь повертай у форматі JSON використовуючи цю схему:\n{format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["title", "content"],
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    pipe = prompt | llm | parser
    return pipe.invoke({
        "title": title,
        "content": content
    })

def load_templates(template_dir="templates"):
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

def load_styles(style_dir="templates/styles"):
    """
    Loads all Markdown templates from the specified directory.
    Returns a dictionary where keys are filenames (without .md) and values are the content.
    """
    styles = {}

    for filename in os.listdir(style_dir):
        if filename.endswith(".html"):
            template_name = os.path.splitext(filename)[0]
            with open(os.path.join(style_dir, filename), "r", encoding="utf-8") as f:
                styles[template_name] = f.read()

    return styles


def generate_summary(title: str, abstract: str, content: dict, template_dir: str):
    TEMPLATES = load_templates(template_dir)
    STYLES = load_styles(os.path.join(template_dir, 'styles'))

    selected_template_name = random.choice(list(TEMPLATES.keys()))
    selected_template = TEMPLATES[selected_template_name]
    selected_style_name = random.choice(list(STYLES.keys()))
    selected_style = STYLES[selected_style_name]

    print(f"Using template: {selected_template_name} and style: {selected_style_name}")

    content.update({"title": title, "abstract": abstract})
    template = selected_template
    for key, value in content.items():
        template = template.replace(f"{{{key}}}", str(value))

    return selected_style.replace('{content}', template), selected_style_name, selected_template


def save_as_pdf(markdown_text, output_dir, filename):
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    raw_html = markdown(markdown_text)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(raw_html)
        page.pdf(path=pdf_path, format="A4", landscape=False, print_background=True)
        browser.close()

    return pdf_path


def save_as_png(markdown_text, output_dir, filename):
    png_path = os.path.join(output_dir, f"{filename}.png")
    raw_html = markdown(markdown_text)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={
                "width": 1280, 
                "height": 800, 
                "deviceScaleFactor": 1
            }
        )
        page.set_content(raw_html)
        page.wait_for_timeout(1000)

        # Get bounding boxes with absolute coordinates
        grounding = find_bounding_box(page)

        # Capture full-page screenshot
        page.screenshot(path=png_path, full_page=True, scale='css')
        browser.close()

    return png_path, grounding

def find_bounding_box(page: Page):
    bounding_boxes = page.evaluate('''
        () => {
            const classNames = [
                "title-container", 
                "table-container", 
                "chart-container", 
                "section-container"
            ];
            
            const typeMapping = {
                "title-container": "text",
                "section-container": "text",
                "table-container": "table",
                "chart-container": "chart"
            };

            // For normalization
            const pageWidth = document.documentElement.scrollWidth;
            const pageHeight = document.documentElement.scrollHeight;

            // Scroll offsets for absolute positioning
            const scrollX = window.scrollX;
            const scrollY = window.scrollY;

            let elements = [];

            classNames.forEach(className => {
                document.querySelectorAll("." + className).forEach(element => {
                    let rect = element.getBoundingClientRect();
                    let type = typeMapping[className] || "text";

                    // Calculate absolute positions by adding scroll offsets
                    let absoluteLeft = rect.left + scrollX;
                    let absoluteTop = rect.top + scrollY;
                    let absoluteRight = rect.right + scrollX;
                    let absoluteBottom = rect.bottom + scrollY;

                    elements.push({
                        "type": type,
                        "content": (type === "text" || type === "chart") 
                            ? element.innerText 
                            : element.innerHTML
                                .replace(/\\s*class="[^"]*"/g, "")
                                .replace(/<\\/?(h1|h2|h3|h4|p)[^>]*>/g, "")
                                .replace(/\\n+/g, " ")
                                .replace(/\\s+/g, " ")
                                .trim(),
                        "box": {
                            "l": absoluteLeft / pageWidth,
                            "t": absoluteTop / pageHeight,
                            "r": absoluteRight / pageWidth,
                            "b": absoluteBottom / pageHeight
                        }
                    });
                });
            });

            // Sort elements by left and top positions
            elements.sort((a, b) => {
                if (a.box.l === b.box.l) {
                    return a.box.t - b.box.t;
                }
                return a.box.l - b.box.l;
            });

            elements.forEach((element, idx) => {
                element.index = idx;
            });

            return elements;
        }
    ''')
    return bounding_boxes



def process_jsonl(llm_model, input_file, output_file, output_dir, template_dir, chunk_size = 10, language = "uk"):
    os.makedirs(output_dir, exist_ok=True)
    llm = ChatOpenAI(model=llm_model, cache=True)
    output_path = os.path.join(output_dir, output_file)

    records = read_jsonl(input_file)
    outputs = []

    for chunk_start in range(0, len(records), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(records))
        chunk = records[chunk_start:chunk_end]

        for i, record in enumerate(chunk, start=chunk_start):
            print(f"Processing record {i+1}/{len(records)}: {record.get('title', 'Unknown Title')}")

            title = record.get("title", "")
            abstract = record.get("abstract", "")
            content = record.get("content", abstract)
            category = record.get("category", "")

            gen_content = generate_content(llm, title, content).model_dump()

            markdown_content, style, template = generate_summary(title, abstract, gen_content, template_dir)

            filename = uuid.uuid4().hex
            png_path, grounding = save_as_png(markdown_content, os.path.join(output_dir, 'images'), filename)

            if style == "scan":
                png_path = dv.distort_cv2(png_path)

            outputs.append(create_output_record(
                lang=language,
                category=category,
                title=title,
                png_path=png_path.removeprefix("dataset/"),
                style=style,
                template=clean_html(template),
                grounding=grounding
            ))

        with open(output_path, "a", encoding="utf-8") as f:
            for output in outputs:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        print(f"Processed {len(outputs)} records. Results saved to {output_path}")
        outputs.clear()
    

def create_output_record(**kwargs):
    output = {
        "lang": kwargs.get("lang", "en"),
        "category": kwargs.get("category", ""),
        "title": kwargs.get("title", ""),
        "image": kwargs.get("png_path", ""),
        "style": kwargs.get("style", ""),
        "template": kwargs.get("template", ""),
        "grounding": kwargs.get("grounding", [])
    }
    return output
   


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
        llm_model="gpt-4o",
        input_file="data/economy_articles_filtered.jsonl",
        output_file="metadata.jsonl",
        output_dir="dataset",
        template_dir="assets/templates",
        chunk_size = 10,
        language = "uk"
    )

    # python src/datagenpipe.py \
    #     --input_file "dataset/economy_articles_filtered.jsonl" \
    #     --output_file "economy_processed.jsonl" \
    #     --llm_model "gpt-4o" \
    #     --output_dir "dataset"
