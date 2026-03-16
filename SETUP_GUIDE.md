# Legal Demand Assistant — Setup Guide

## Overview

This app compares 4 domain adaptation configurations for a legal demand assistant built on Qwen3.5-9B:

| Config | Model | Prompt | Inference |
|---|---|---|---|
| **Baseline** | Base Qwen3.5-9B | Generic prompt | Live (HuggingFace API) |
| **GEPA** | Base Qwen3.5-9B | GEPA-optimized prompts | Live (HuggingFace API) |
| **QLoRA** | Fine-tuned (LoRA adapter) | Generic prompt | Precomputed (from A100 eval) |
| **GEPA + QLoRA** | Fine-tuned (LoRA adapter) | GEPA-optimized prompts | Precomputed (from A100 eval) |

Baseline and GEPA configs call the HuggingFace Inference API in real time. QLoRA configs require GPU inference, so they serve precomputed responses from the Colab evaluation run (`evaluation/eval_results.json`).

---

## Prerequisites

- Python 3.10+
- A free HuggingFace account and API token

---

## Step 1: Clone the Repo

```bash
git clone https://github.com/BigDataAnalytics-CS5542/Lab_8.git
cd Lab_8
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `fastapi`, `uvicorn`, `httpx`, `streamlit`, `pydantic`, `requests`

---

## Step 3: Get a HuggingFace API Token

1. Go to https://huggingface.co/settings/tokens
2. Click "Create new token"
3. Name it anything (e.g., "lab8")
4. Select "Read" access (free tier is fine)
5. Copy the token (starts with `hf_...`)

---

## Step 4: Set Environment Variable

```bash
export HF_TOKEN="hf_your_token_here"
```

On Windows (PowerShell):
```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

---

## Step 5: Start the Backend

```bash
uvicorn backend.app:app --port 3001
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:3001
```

Verify it's working by opening http://localhost:3001 in a browser — you should see:
```json
{"status": "ok", "model": "Qwen/Qwen3.5-9B", "configs": ["baseline", "gepa", "qlora", "gepa_qlora"]}
```

---

## Step 6: Start the Frontend (New Terminal)

Open a **new terminal window/tab** (keep the backend running), then:

```bash
cd Lab_8
export HF_TOKEN="hf_your_token_here"
streamlit run frontend/app.py --server.port 3000
```

The app will open at http://localhost:3000

---

## Using the App

### Single Config Mode
1. Select a configuration from the sidebar dropdown
2. Type or paste a legal query
3. Click "Submit"
4. View the response with metadata (live vs precomputed, task type, score)

### Compare All Mode
1. Switch to "Compare All" in the sidebar
2. Enter a query
3. Click "Submit"
4. See all 4 configs side-by-side in a 2x2 grid

### Sample Queries
The sidebar includes clickable sample queries that match the 10 evaluation queries. These are the best for demonstrating the comparison since precomputed results are available for all 4 configs.

### Evaluation Results
Click "View Full Evaluation Results" at the bottom to see the per-query and per-task score tables.

---

## Project Structure

```
Lab_8/
├── backend/
│   └── app.py                          # FastAPI backend (4-config toggle)
├── frontend/
│   └── app.py                          # Streamlit UI (chat + comparison)
├── training/
│   ├── qlora_finetune.ipynb            # QLoRA fine-tuning notebook (run on Colab)
│   ├── qlora_finetune.py               # QLoRA script version
│   ├── gepa_prompt_optimization.ipynb  # GEPA optimization notebook (run on Colab)
│   └── gepa_optimized_prompts.json     # GEPA output (optimized prompts per task)
├── evaluation/
│   ├── evaluation_queries.md           # 10 held-out evaluation queries
│   ├── eval_4_configs.ipynb            # 4-config evaluation notebook (run on Colab)
│   ├── eval_results.json               # Full evaluation results (responses + scores)
│   └── results.md                      # Results summary with analysis
├── data/
│   └── instruction_dataset.json        # 50-example training dataset
├── requirements.txt                    # Python dependencies
└── SETUP_GUIDE.md                      # This file
```

---

## Key Files

| File | Purpose |
|---|---|
| `training/gepa_optimized_prompts.json` | GEPA-evolved prompts used by the GEPA and GEPA+QLoRA configs |
| `evaluation/eval_results.json` | Precomputed responses from all 4 configs (generated on A100 GPU) |
| `data/instruction_dataset.json` | 50 instruction/input/output examples used for QLoRA fine-tuning |

---

## Troubleshooting

**"Backend not running" error in Streamlit**
- Make sure the backend is running in a separate terminal: `uvicorn backend.app:app --port 3001`

**"HF_TOKEN not set" error**
- Set the environment variable: `export HF_TOKEN="hf_..."`
- Make sure it's set in both terminals (backend and frontend)

**HuggingFace API returns 401/403**
- Check that your token is valid at https://huggingface.co/settings/tokens
- Free tier tokens work fine for inference

**HuggingFace API is slow or times out**
- The free tier can be slow (30-60s per request). This is normal for a 9B model.
- If it consistently times out, try again later — HF free tier has rate limits.

**QLoRA responses say "No precomputed response found"**
- QLoRA configs only have precomputed responses for the 10 evaluation queries
- Use the sample queries in the sidebar for best results
- For custom queries, use Baseline or GEPA configs (live inference)

---

## Reproducing the Training (Optional)

### QLoRA Fine-Tuning
1. Open `training/qlora_finetune.ipynb` in Google Colab
2. Set runtime to GPU (A100 recommended)
3. Run all cells
4. Download the adapter zip from the last cell

### GEPA Prompt Optimization
1. Open `training/gepa_prompt_optimization.ipynb` in Google Colab
2. Set runtime to GPU
3. Run all cells
4. Download `gepa_optimized_prompts.json` from the last cell

### 4-Config Evaluation
1. Open `evaluation/eval_4_configs.ipynb` in Google Colab
2. Upload the QLoRA adapter zip and GEPA prompts JSON
3. Run all cells
4. Download `eval_results.json` and `results.md`
