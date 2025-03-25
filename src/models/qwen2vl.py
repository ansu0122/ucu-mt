import torch
from PIL import Image
from typing import Optional, Union, List, Mapping, Any
from tqdm import tqdm
import json

from pydantic import BaseModel, Field, ValidationError
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration


class AnswerSchema(BaseModel):
    text: str = Field(description="Заголовки і вміст текстових розділів документу")
    titles: list = Field(description="Список заголовків текстових розділів, таблиці та графіку")
    table: str = Field(description="Таблиця з даними в форматі HTML")


def parse_document_response(output_text: str) -> Optional[AnswerSchema]:
    try:
        cleaned = output_text.strip()
        if not cleaned.startswith("{"):
            cleaned = cleaned[cleaned.find("{"):]
        data = json.loads(cleaned)
        return AnswerSchema(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[Parsing Error] Could not parse:\n{output_text}\n\nError: {e}")
        return None


def build_doc_extraction_prompt() -> str:
    return (
        "Витягни заголовки і текст розділів документу в порядку їх прочитання.\n"
        "Витягни таблицю з даними в форматі HTML.\n\n"
        "Поверни результат у форматі JSON з такими полями:\n"
        "{\n"
        '  "text": "Повний текст з заголовками і вмістом",\n'
        '  "titles": ["Заголовок 1", "Заголовок 2", "Таблиця", "Графік"],\n'
        '  "table": "<table>...</table>"\n'
        "}"
    )


class QwenVL2_LLM:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", max_new_tokens=1024, device="cuda"):
        self.max_new_tokens = max_new_tokens
        self.device = device

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def generate(self, prompt: str, image: Image.Image) -> str:
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return output_text[0] if output_text else ""


def process_doc_image(image: Image, model: QwenVL2_LLM) -> Optional[AnswerSchema]:
    image = image.convert("RGB")
    prompt = build_doc_extraction_prompt()
    response = model.generate(prompt=prompt, image=image)
    return parse_document_response(response)
