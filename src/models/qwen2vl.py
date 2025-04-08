import re
import torch
from PIL import Image
from typing import Optional, Union, List, Mapping, Any
from tqdm import tqdm
import json

from pydantic import BaseModel, Field, ValidationError
from unsloth import FastVisionModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


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
            attn_implementation="eager"
        )
        self.model = self.model.to(dtype=torch.bfloat16)

        self.tokenizer.use_fast = True

        # self.model.to(device)
        FastVisionModel.for_inference(self.model)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def generate(self, image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt}
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
            # temperature=1.0,
            # min_p = 0.1
        )

        output = self.model.generate(**inputs, **gen_args)
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded = "".join(decoded)
        return decoded if decoded else ""


    def process_doc_image(self, image: Image): #-> Optional[AnswerSchema]:
        torch.cuda.empty_cache()

        image = image.convert("RGB")
        response = self.generate(image=image)

        if isinstance(response, list):
            response = response[0]

        if "assistant\n" in response:
            response = response.split("assistant\n", 1)[1].strip()
        if "<table>" in response:
            response = re.search(r"<table>.*?</table>", response, re.DOTALL)
            response = response.group(0) if response else ""
            
        # return parse_document_response(parser, response)
        return response


