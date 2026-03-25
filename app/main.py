import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from app.graph import get_graph  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="LangGraph Router Demo")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    route: str
    response: str


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html missing")
    return FileResponse(index_path)


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
        )
    try:
        result = get_graph().invoke(
            {"prompt": body.prompt.strip(), "route": "", "response": ""}
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(route=result["route"], response=result["response"])
