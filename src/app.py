import os
from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from models import qwen2vl as qwn
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = qwn.QwenVL2_LLM(
    model_name = "ansu0122/uadoc-ada-qwen2.5vl",
    max_new_tokens = 4096,
    device = "cuda",
    load_in_4bit = False
)

@app.post("/generate")
async def generate(file: UploadFile, prompt: str = Form(...), authorization: str = Header(None)):

    api_key = os.getenv("FAST_API_KEY")
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image upload")

    model.set_prompt(prompt)
    response = model.process_doc_image(image)

    return {"response": response}
