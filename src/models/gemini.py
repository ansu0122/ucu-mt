import os
from typing import Union
from pathlib import Path
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiFlashModel:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
    ):
        self.model_name = model_name
        self.prompt = None
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def generate(self, image: Union[str, Path, Image.Image]) -> str:
        # Convert string path to PIL.Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        assert self.prompt is not None, "Prompt is not set. Use set_prompt(prompt) before generate()."

        # Send prompt and image to Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.prompt, image]
        )

        return response.text.strip() if hasattr(response, "text") else ""

    def process_doc_image(self, image: Union[str, Path, Image.Image]) -> str:
        try:
            return self.generate(image=image)
        except Exception as e:
            print(f"Error processing image with Gemini: {e}")
            return None


if __name__ == "__main__":

    query = """
        Витягни текст зображення у точному вигляді.
        – Залиши слова, літери й розділові знаки такими, як вони є.
        – Збережи порядок, великі й малі літери, пробіли, розриви рядків.
        – Будь-яке редагування, переклад, дописування або зміна заборонені.
        - Повертай таблицю в Markdown-форматі.

        Поверни лише текст — дослівно.
    """

    ocr_model = GeminiFlashModel(model_name = "gemini-2.0-flash")
    ocr_model.set_prompt(query)
    result = ocr_model.process_doc_image("dataset/images/46f856e3f0ff4afabbf15754eea3340f.jpg")
    print(result)