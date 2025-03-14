from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

class QwenLLM(LLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model.name_or_path}

def down_load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return QwenLLM(model, tokenizer)


def call_inference_pipe(llm, prompt):
    return llm(prompt)