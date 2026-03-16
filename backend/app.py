"""
Legal Demand Assistant — FastAPI Backend
=========================================
Serves 4 configurations:
  - Baseline: HuggingFace API + generic prompt (live)
  - GEPA: HuggingFace API + optimized prompts (live)
  - QLoRA: Precomputed responses from eval run
  - GEPA + QLoRA: Precomputed responses from eval run
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="Legal Demand Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-9B"
HF_API_URL = f"https://router.huggingface.co/novita/v3/openai/chat/completions"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

BASELINE_PROMPT = (
    "You are a legal demand assistant. You help draft demand letters, "
    "identify legal claims, extract elements from demand letters, "
    "evaluate letter quality, and recommend remedies."
)

# Load GEPA-optimized prompts
GEPA_PROMPTS_PATH = Path(__file__).parent.parent / "training" / "gepa_optimized_prompts.json"
if GEPA_PROMPTS_PATH.exists():
    with open(GEPA_PROMPTS_PATH) as f:
        gepa_data = json.load(f)
    GEPA_PROMPTS = gepa_data["optimized_prompts"]
else:
    GEPA_PROMPTS = {}

# Load precomputed results for QLoRA configs
EVAL_RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "eval_results.json"
if EVAL_RESULTS_PATH.exists():
    with open(EVAL_RESULTS_PATH) as f:
        EVAL_RESULTS = json.load(f)
    PRECOMPUTED = EVAL_RESULTS.get("detailed_results", {})
else:
    PRECOMPUTED = {}

# Task type detection keywords
TASK_KEYWORDS = {
    "draft_demand_letter": ["draft", "demand letter", "write a letter", "prepare a letter"],
    "identify_claim": ["identify", "claim", "classify", "legal claim", "strongest claim"],
    "extract_elements": ["extract", "elements", "claimant", "recipient", "damages"],
    "evaluate_letter": ["evaluate", "assess", "complete", "effective", "adequate"],
    "recommend_remedy": ["remedy", "remedies", "suggest", "recommend", "appropriate"],
}


def detect_task_type(user_input: str) -> str:
    """Detect task type from user input."""
    input_lower = user_input.lower()
    best_match = "draft_demand_letter"
    best_count = 0
    for task_type, keywords in TASK_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in input_lower)
        if count > best_count:
            best_count = count
            best_match = task_type
    return best_match


# ── Request/Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    config: str = "baseline"  # baseline, gepa, qlora, gepa_qlora


class QueryResponse(BaseModel):
    response: str
    config: str
    config_label: str
    task_type: str
    is_live: bool  # True if live inference, False if precomputed
    score: float | None = None


# ── HuggingFace API Call ─────────────────────────────────────────────────────

async def call_hf_api(system_prompt: str, user_input: str) -> str:
    """Call HuggingFace Inference API for Qwen3.5-9B."""
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN not set. Set the HF_TOKEN environment variable.",
        )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(HF_API_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"HuggingFace API error: {resp.text}",
        )

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def find_precomputed(config_key: str, user_input: str) -> dict | None:
    """Search precomputed results for a matching query."""
    results = PRECOMPUTED.get(config_key, [])
    input_lower = user_input.lower()

    # Try exact substring match on the query input
    for r in results:
        # Check if the eval query text appears in the user input or vice versa
        if any(
            keyword in input_lower
            for keyword in [
                r.get("query_id", "").lower(),
                r.get("response", "")[:20].lower(),
            ]
        ):
            return r

    # Fuzzy match: find the most similar precomputed query by keyword overlap
    best_match = None
    best_overlap = 0
    for r in results:
        response_words = set(r.get("response", "").lower().split()[:20])
        input_words = set(input_lower.split())
        # Match based on task type similarity
        task_type = detect_task_type(user_input)
        if r.get("task_type") == task_type:
            overlap = len(response_words & input_words)
            if overlap > best_overlap or best_match is None:
                best_overlap = overlap
                best_match = r

    return best_match


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {"status": "ok", "model": MODEL_ID, "configs": ["baseline", "gepa", "qlora", "gepa_qlora"]}


@app.get("/configs")
async def get_configs():
    """Return available configurations and their descriptions."""
    return {
        "configs": [
            {
                "key": "baseline",
                "label": "Baseline",
                "description": "Base Qwen3.5-9B with generic prompt",
                "is_live": True,
            },
            {
                "key": "gepa",
                "label": "GEPA",
                "description": "Base model + GEPA-optimized prompts",
                "is_live": True,
            },
            {
                "key": "qlora",
                "label": "QLoRA",
                "description": "Fine-tuned model (precomputed on A100)",
                "is_live": False,
            },
            {
                "key": "gepa_qlora",
                "label": "GEPA + QLoRA",
                "description": "Fine-tuned model + optimized prompts (precomputed on A100)",
                "is_live": False,
            },
        ]
    }


@app.get("/eval-summary")
async def eval_summary():
    """Return the evaluation summary from the 4-config comparison."""
    if not EVAL_RESULTS:
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    return EVAL_RESULTS.get("summary", {})


@app.get("/eval-results")
async def eval_results():
    """Return full evaluation results."""
    if not EVAL_RESULTS:
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    return EVAL_RESULTS


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run a query against the selected configuration."""
    task_type = detect_task_type(req.query)
    config_labels = {
        "baseline": "Baseline",
        "gepa": "GEPA",
        "qlora": "QLoRA",
        "gepa_qlora": "GEPA + QLoRA",
    }

    if req.config not in config_labels:
        raise HTTPException(status_code=400, detail=f"Invalid config: {req.config}")

    # QLoRA configs use precomputed responses
    if req.config in ("qlora", "gepa_qlora"):
        match = find_precomputed(req.config, req.query)
        if match:
            return QueryResponse(
                response=match["response"],
                config=req.config,
                config_label=config_labels[req.config],
                task_type=match.get("task_type", task_type),
                is_live=False,
                score=match.get("total_score"),
            )
        else:
            return QueryResponse(
                response=(
                    f"No precomputed response found for this query. "
                    f"QLoRA responses require GPU inference (run on Colab with A100). "
                    f"Try one of the 10 evaluation queries for precomputed results."
                ),
                config=req.config,
                config_label=config_labels[req.config],
                task_type=task_type,
                is_live=False,
            )

    # Baseline and GEPA use live HuggingFace API
    if req.config == "gepa":
        system_prompt = GEPA_PROMPTS.get(task_type, BASELINE_PROMPT)
    else:
        system_prompt = BASELINE_PROMPT

    response = await call_hf_api(system_prompt, req.query)

    return QueryResponse(
        response=response,
        config=req.config,
        config_label=config_labels[req.config],
        task_type=task_type,
        is_live=True,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
