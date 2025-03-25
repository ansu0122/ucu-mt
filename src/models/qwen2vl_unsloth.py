import torch
from PIL import Image
from typing import Optional, Union, List, Mapping, Any
from tqdm import tqdm
import json

from pydantic import BaseModel, Field, ValidationError
from unsloth import FastVisionModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


class AnswerSchema(BaseModel):
    text: str = Field(description="Заголовки і вміст текстових розділів документу")
    titles: list = Field(description="Список заголовків текстових розділів, таблиці та графіку")
    table: str = Field(description="Таблиця з даними в форматі HTML")


def parse_document_response(parser: PydanticOutputParser, output_text: str) -> Optional[AnswerSchema]:
    try:
        return parser.parse(output_text)
    except Exception as e:
        print(f"[Parsing Error] Could not parse:\n{output_text}\n\nError: {e}")
        return None


def build_doc_extraction_prompt(parser: PydanticOutputParser) -> str:
    prompt_template = PromptTemplate(
        template=(
            "Витягни заголовки і текст розділів документу в порядку їх прочитання.\n"
            "Витягни таблицю з даними в форматі HTML.\n\n"
            "Поверни результат у форматі JSON згідно з цією схемою:\n"
            "{format_instructions}"
        ),
        input_variables=[],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt_template.format()


class QwenVL2_LLM:
    def __init__(
        self,
        model_name: str = "unsloth/Qwen2-VL-7B-Instruct",
        max_new_tokens: int = 1024,
        device: str = "cuda",
        load_in_4bit: bool = True,
        use_gradient_checkpointing: Union[bool, str] = "unsloth"
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device

        # Load model + tokenizer via Unsloth
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.model.to(device)
        FastVisionModel.for_inference(self.model)

    def generate(self, prompt: str, image: Union[str, Image.Image], **generation_args) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)

        gen_args = dict(
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            temperature=1.0,
            do_sample=False,
            **generation_args
        )

        output = self.model.generate(**inputs, **gen_args)
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return decoded[0] if decoded else ""


def process_doc_image(image: Image, model: QwenVL2_LLM) -> Optional[AnswerSchema]:
    image = image.convert("RGB")
    parser = PydanticOutputParser(pydantic_object=AnswerSchema)
    prompt = build_doc_extraction_prompt(parser)
    response = model.generate(prompt=prompt, image=image)
    return parse_document_response(parser, response)


if __name__ == "__main__":

    image_path = "your/image/path.jpg"
    image = Image.open(image_path).convert("RGB")

    model = QwenVL2_LLM()
    result = process_doc_image(image, model)

    if result:
        print("Titles:", result.titles)
        print("Text Preview:", result.text[:300])
        print("Table HTML:", result.table[:300])
    else:
        print("Failed to parse the response.")