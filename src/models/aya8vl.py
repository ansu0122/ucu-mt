import torch
from PIL import Image
from typing import Union, List, Dict

from transformers import AutoProcessor, AutoModelForImageTextToText


class AyaVisionLLM:
    def __init__(
        self,
        model_name: str = "CohereForAI/aya-vision-8b",
        max_new_tokens: int = 300,
        device: str = "cuda",
        dtype=torch.float16,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map=device, torch_dtype=dtype
        )
        self.device = self.model.device
        self.prompt: str = ""

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def generate(self, image: Union[str, Image.Image], prompt: str = None) -> str:
        if prompt is not None:
            self.set_prompt(prompt)
        if not self.prompt:
            raise ValueError("Prompt is not set. Use `set_prompt()` or pass `prompt` to `generate()`.")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        messages: List[Dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
        )

        decoded = self.processor.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decoded = "".join(decoded)

        return decoded if decoded else ""

    def process_doc_image(self, image: Union[str, Image.Image]) -> str:
        torch.cuda.empty_cache()
        return self.generate(image=image)
