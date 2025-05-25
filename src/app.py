import os
import re
from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model, tokenizer = FastVisionModel.from_pretrained(
    "ansu0122/uadoc-ada-qwen2.5vl_exp2",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

@app.post("/generate")
async def generate(file: UploadFile,
                   authorization: str = Header(None),
                   prompt: str = Form(...),
                   temperature: float = Form(0.001),
                   max_new_tokens: int = 4096,
                   min_p: float = 0.1):

    api_key = os.getenv("FAST_API_KEY")
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image upload")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    output = model.generate(**inputs, max_new_tokens = max_new_tokens,
                    use_cache = True, temperature = temperature, min_p = min_p)

    decoded = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = "".join(decoded)

    if "assistant\n" in decoded:
        decoded = decoded.split("assistant\n", 1)[1].strip()
    if "<table>" in decoded:
        decoded = re.search(r"<table>.*?</table>", decoded, re.DOTALL)
        decoded = decoded.group(0) if decoded else ""

    return {"response": decoded}
