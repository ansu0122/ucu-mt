import re
import torch
from PIL import Image
from typing import Union
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from flash_attn.bert_padding import index_first_axis
import transformers.models.phi.modeling_phi as phi_model
if not hasattr(phi_model, "index_first_axis"):
    setattr(phi_model, "index_first_axis", index_first_axis)

class Phi4VisionLLM:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        max_new_tokens: int = 2048,
        device: str = "cuda",
        attn_mech: str = "flash_attention_2"
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation=attn_mech,
        ).to(device)

        self.generation_config = GenerationConfig.from_pretrained(model_name)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def generate(self, image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        full_prompt = f"{self.user_prompt}<|image_1|>{self.prompt}{self.prompt_suffix}{self.assistant_prompt}"

        inputs = self.processor(text=full_prompt, images=image, return_tensors='pt').to(self.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            num_logits_to_keep=self.max_new_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # Remove the prompt part from the output
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        decoded = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decoded = "".join(decoded)

        return decoded if decoded else ""

    def process_doc_image(self, image: Image.Image) -> str:
        torch.cuda.empty_cache()
        return self.generate(image=image)
