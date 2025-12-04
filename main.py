from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="GPT Text Generation Service")

# Uses OPENAI_API_KEY env var (set in Kubernetes)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

client = OpenAI()  # the SDK will read OPENAI_API_KEY from env


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = "gpt-4.1-mini"  # light model
    max_tokens: int | None = 128        # keep responses small


class GenerateResponse(BaseModel):
    text: str


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(body: GenerateRequest):
    try:
        response = client.chat.completions.create(
            model=body.model,
            messages=[{"role": "user", "content": body.prompt}],
            max_tokens=body.max_tokens,
        )
        text = response.choices[0].message.content
        return GenerateResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

