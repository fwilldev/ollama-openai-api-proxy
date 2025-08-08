import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from openai import OpenAI, BaseModel, OpenAIError, NOT_GIVEN
from openai.types.shared_params import Reasoning

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

app = FastAPI(title="Ollama to OpenAI Proxy Service")

# use official OpenAI Client
client = OpenAI(
    api_key=os.environ.get(OPENAI_API_KEY),
)

# if desired, expand with additional models accordingly.
ALLOWED_MODELS = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "o1-mini",
    "o4",
    "o4-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano"
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatResponseRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    think: Optional[bool] = False

@dataclass
class TagsResponse:
    name: str
    model: str

def is_reasoning_model(model):
    return model in ['gpt-4o', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4o-mini', 'o4-mini', 'o3', 'o3-pro', 'o3-mini', 'o1', 'o1-pro', 'o1-mini']

def now_rfc3339() -> str:
    return datetime.now().isoformat("T") + "Z"

def is_gpt_5(model):
    return model in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano']

@app.post("/api/chat")
async def chat_completions(req: ChatResponseRequest):
    """
        Handels chat completion requests and proxies them to OpenAI.

        Args:
            req (ChatResponseRequest): A request object containing the
            necessary parameters, including:

            - model (str): The model name to use for generating responses.
            - messages (list): A list of messages in the chat history.
            - temperature (float, optional): Controls the randomness of responses.
            - stream (bool, optional): If true, the response will be streamed.
            - think (bool, optional): Allows for processing before responding.
        Returns:
            JSON response containing the generated chat completion. Ollama Client compatible.
    """
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    reasoning_arg = Reasoning(effort="medium") if req.think and is_reasoning_model(req.model) else NOT_GIVEN
    try:
        if req.stream:
            streamer = client.responses.create(
                model=req.model,
                input=prompt,
                temperature=req.temperature if not is_gpt_5(req.model) else NOT_GIVEN,
                stream=True,
                reasoning=reasoning_arg
            )
        def event_generator():
            last_finish = "stop"
            for evt in streamer:
                ed = evt.model_dump()
                etype = ed.get("type")
                if etype == "response.output_text.delta":
                    delta = ed.get("delta", "")
                    frame = {
                        "model":      req.model,
                        "created_at": now_rfc3339(),
                        "message":    {"role": "assistant", "content": delta},
                        "done":       False,
                    }
                    yield json.dumps(frame) + "\n"
                if etype == "response.completed":
                    break

            final = {
                "model":             req.model,
                "created_at":        now_rfc3339(),
                "message":           {"role": "assistant", "content": ""},
                "done":              True,
                "finish_reason":     last_finish,
                "total_duration":    0,
                "load_duration":     0,
                "prompt_eval_count": 0,
                "eval_count":        0,
                "eval_duration":     0,
            }
            yield json.dumps(final) + "\n"

        return StreamingResponse(
            event_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection":    "keep-alive",
            },
        )
    except OpenAIError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/tags")
def get_tags():
    """
    Ollama‐compatible tags endpoint
    """
    tags: list[TagsResponse] = []
    try:
        for m in ALLOWED_MODELS:
            tags.append(TagsResponse(m, m))

        tags_dicts = [asdict(t) for t in tags]
        return JSONResponse(status_code=200, content={"models": tags_dicts})
    except OpenAIError as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return "Ollama is running"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
