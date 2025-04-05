import os
import time
import base64
from pathlib import Path
from typing import Union, Type
from PIL import Image
from pydantic import BaseModel
from mistralai import Mistral
from mistralai import ImageURLChunk, TextChunk
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


class MistralOCRModel:
    def __init__(
        self,
        ocr_model: str = "mistral-ocr-latest",
        chat_model: str = "pixtral-12b-latest",
        temperature: float = 0.0,
    ):
        self.ocr_model = ocr_model
        self.chat_model = chat_model
        self.temperature = temperature
        self.schema: Type[BaseModel] = None
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    def set_output_schema(self, schema_model: Type[BaseModel]):
        """
        Set the expected output schema for the structured OCR.
        """
        self.schema = schema_model

    def set_prompt(self, prompt: str):
        """
        Set the prompt for the chat model.
        """
        self.prompt = prompt

    def generate(self, image: Union[str, Path, Image.Image]) -> BaseModel:
        assert self.schema is not None, "Output schema not set. Use set_output_schema(schema) before generate."
        if isinstance(image, Image.Image):
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                image_path = tmp.name
        else:
            image_path = str(image)

        image_file = Path(image_path)
        assert image_file.is_file(), "The provided image path does not exist."
        encoded_image = base64.b64encode(image_file.read_bytes()).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

        # Step 1: OCR
        image_response = self.client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model=self.ocr_model
        )
        ocr_markdown = image_response.pages[0].markdown
        time.sleep(2.0)

        # Step 2: Structured parsing via chat
        chat_response = self.client.chat.parse(
            model=self.chat_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        ImageURLChunk(image_url=base64_data_url),
                        TextChunk(text=(
                            f"""
                                Це OCR зображення у форматі Markdown:\n{ocr_markdown}\n.\n
                                {self.prompt}
                            """
                        ))
                    ]
                }
            ],
            response_format=self.schema,
            temperature=self.temperature
        )

        return chat_response.choices[0].message.parsed

    def process_doc_image(self, image: Union[str, Path, Image.Image]) -> BaseModel:
        return self.generate(image=image)


if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from typing import List, Dict

    class TitleSchema(BaseModel):
        titles: list[str] = Field(description="Назви для усіх розділів документу, таблиці і графіку в порядку прочитання.")

    query = """
        Витягни назви всіх розділів документа, таблиць і графіків у порядку, в якому вони з’являються в документі.

        Вимоги:
        – Витягни лише ті елементи, які виглядають як заголовки, підписи або назви секцій документу.
        – Залиш назви у тому вигляді, в якому вони подані у документі.
        – Збережи точний порядок прочитання — зверху вниз, зліва направо.
        – Уникай інтерпретацій, описів чи узагальнень.
        – Поверни тільки список назв без додаткового тексту.

        Важливо:
        Текст кожної назви має залишатися без змін. Будь-які правки або перекручення заборонено.
    """

    ocr_model = MistralOCRModel()
    ocr_model.set_prompt(query)
    ocr_model.set_output_schema(TitleSchema)
    result = ocr_model.process_doc_image("dataset/images/0a2dbad583cd485f885994f5294d2ae8.jpg")
    print(result)
