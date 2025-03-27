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
    # text: str = Field(description="Заголовки і вміст текстових розділів документу")
    # titles: list = Field(description="Список заголовків текстових розділів, таблиці та графіку")
    table: str = Field(description="Таблиця з даними в форматі HTML")


def parse_document_response(parser: PydanticOutputParser, output_text: str) -> Optional[AnswerSchema]:
    try:
        return parser.parse(output_text)
    except Exception as e:
        print(f"[Parsing Error] Could not parse:\n{output_text}\n\nError: {e}")
        return None


def build_doc_extraction_prompt(parser: PydanticOutputParser) -> str:
    # prompt_template = PromptTemplate(
    #     template=(
    #         # "Витягни заголовки і текст розділів документу в порядку їх прочитання.\n"
    #         "Витягни таблицю з даними в форматі HTML.\n\n"
    #         "Поверни результат у форматі JSON згідно з цією схемою:\n"
    #         "{\"table\":\"<table>...</table>\"}"
    #     ),
    #     input_variables=[],
    #     partial_variables={"format_instructions": parser.get_format_instructions()}
    # )
    # return prompt_template.format()

    return (
        "Витягни таблицю з даними в форматі HTML.\n\n"
        "Поверни результат у форматі JSON з такими полями:\n"
        "{\n"
        '  "table": "<table>...</table>"\n'
        "}"
    )


class QwenVL2_LLM:
    def __init__(
        self,
        model_name: str = "unsloth/Qwen2-VL-7B-Instruct",
        max_new_tokens: int = 2048,
        device: str = "cuda",
        load_in_4bit: bool = True,
        use_gradient_checkpointing: Union[bool, str] = "unsloth"
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.tokenizer.use_fast = True

        # self.model.to(device)
        FastVisionModel.for_inference(self.model)

    def generate(self, prompt: str, image: Union[str, Image.Image]) -> str:
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
            temperature=1.5,
            min_p = 0.1
        )

        output = self.model.generate(**inputs, **gen_args)
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return decoded if decoded else ""


def process_doc_image(image: Image, model: QwenVL2_LLM): #-> Optional[AnswerSchema]:
    image = image.convert("RGB")
    parser = PydanticOutputParser(pydantic_object=AnswerSchema)
    prompt = build_doc_extraction_prompt(parser)
    response = model.generate(prompt=prompt, image=image)

    if isinstance(response, list):
        response = response[0]

    if "assistant\n" in response:
        response = response.split("assistant\n", 1)[1].strip()
        
    # return parse_document_response(parser, response)
    return response

    