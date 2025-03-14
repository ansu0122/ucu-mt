import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Optional, List, Mapping, Any, Union

model_name = "Qwen/Qwen-VL-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

class QwenVL2_LLM(LLM):
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

    @property
    def _llm_type(self) -> str:
        return "Qwen-VL2"

    def _call(self, prompt: str, image: Optional[Union[str, Image.Image]] = None, stop: Optional[List[str]] = None) -> str:
        if isinstance(image, str):
            image = Image.open(image)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model.name_or_path}


llm = QwenVL2_LLM(model, tokenizer, processor)

def call_inference_pipe(image_path: str, question: str) -> str:
    """
    Calls the Qwen-VL2 model with an image and a text prompt.

    :param image_path: Path to the image file
    :param question: Text prompt for the model
    :return: Model's response
    """

    response = llm._call(prompt=question, image=image_path)
    print("Response (Direct Model Call):", response)

    prompt = PromptTemplate.from_template("Analyze this image and answer: {question}")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response_chain = chain.run({"question": question})
    print("Response (LangChain Pipeline):", response_chain)

    return response_chain

if __name__ == "__main__":
    image_path = "../../dataset/Економіка_1.png"
    question = "What is in this image?"
    call_inference_pipe(image_path, question)
