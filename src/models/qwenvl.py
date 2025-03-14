import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any, Union

class QwenVL2_LLM(LLM):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", max_new_tokens: int = 512):
        """
        Initializes the Qwen-VL model for vision-language processing.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    @property
    def _llm_type(self) -> str:
        return "Qwen2-VL"

    def _call(self, prompt: str, image: Optional[Union[str, Image.Image]] = None, stop: Optional[List[str]] = None) -> str:
        """
        Generates a response based on a text prompt and an optional image.

        :param prompt: User's question or instruction
        :param image: Path to an image or a PIL image object
        :return: Generated response from the model
        """

        if isinstance(image, str):
            image = Image.open(image)

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(self.device)

        self.model.to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return output_text[0] if output_text else ""

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model.name_or_path}


def call_inference_pipe(llm: LLM, image_path: str, question: str) -> str:
    """
    Calls the Qwen-VL2 model with an image and a text prompt.

    :param image_path: Path to the image file
    :param question: Text prompt for the model
    :return: Model's response
    """
    response = llm._call(prompt=question, image=image_path)
    
    return response


if __name__ == "__main__":
    llm = QwenVL2_LLM()

    image_path = "../../dataset/Економіка_1.png"
    question = "Витягни таблицю із зображення в HTML форматі?"

    response = call_inference_pipe(llm, image_path, question)
