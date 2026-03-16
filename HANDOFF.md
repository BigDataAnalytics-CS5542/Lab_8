# Lab 8 Handoff — What's Done and What's Left

## What's Been Completed

### 1. Instruction Dataset (Step 1 & 2) — DONE
- 50 examples in `data/instruction_dataset.json`
- 5 task types: demand letter drafting, claim identification, element extraction, letter evaluation, remedy recommendation
- 10 held-out evaluation queries in `evaluation/evaluation_queries.md`

### 2. QLoRA Fine-Tuning (Step 3, Option A) — DONE
- Qwen3.5-9B fine-tuned on Colab Pro (A100 GPU)
- Training: 3 epochs, ~2.7 minutes, loss 0.85
- Adapter weights saved (116 MB) — not in repo due to size, available on request
- Notebook: `training/qlora_finetune.ipynb`

### 3. GEPA Prompt Optimization (Step 3, Option B) — DONE
- Autonomous prompt evolution using GEPA methodology
- 3 generations of reflect → mutate → evaluate → select
- Improved from 58% → 88% on eval queries
- Optimized prompts saved: `training/gepa_optimized_prompts.json`
- Notebook: `training/gepa_prompt_optimization.ipynb`

### 4. 4-Config Evaluation (Step 5) — DONE
- All 10 eval queries run against all 4 configs on Colab (A100)
- Results: Baseline 90%, GEPA 89%, QLoRA 95%, GEPA+QLoRA 92%
- Full results: `evaluation/eval_results.json`
- Analysis: `evaluation/results.md`
- Notebook: `evaluation/eval_4_configs.ipynb`

### 5. FastAPI + Streamlit App (Step 4 & 6) — DONE
- Backend: `backend/app.py`
- Frontend: `frontend/app.py`
- Baseline & GEPA use live HuggingFace API
- QLoRA configs use precomputed responses from eval run
- Side-by-side comparison mode

---

## What's Left

### 1. Test the App Locally
Follow `SETUP_GUIDE.md` to verify everything works:

```bash
# Install dependencies
pip install -r requirements.txt

# Get a free HuggingFace token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Terminal 1 — start backend
uvicorn backend.app:app --port 3001

# Terminal 2 — start frontend
streamlit run frontend/app.py --server.port 3000
```

Open http://localhost:3000, try "Compare All" mode with a sample query from the sidebar.

### 2. Group Report (PDF, 1-2 pages)
Compile a PDF covering these sections. All data is already in the repo:

| Section | Source |
|---|---|
| Project title and team members | Fill in |
| Domain task definition | `LAB8_OVERVIEW.md` (lines 62-79) |
| Instruction dataset description | 50 examples, 5 task types, JSON format. See `data/instruction_dataset.json` |
| Adaptation methods used | QLoRA fine-tuning (Qwen3.5-9B, LoRA r=16, 3 epochs) + GEPA prompt optimization (3 generations, autonomous reflect-mutate loop) |
| System integration | FastAPI backend + Streamlit UI with 4-config toggle. Live inference via HF API for base model, precomputed for QLoRA. |
| Evaluation results | Copy tables from `evaluation/results.md` |
| Impact on performance | Baseline 90% → QLoRA 95% (+5%). See "Observations" in `evaluation/results.md` |

#### Contribution Table

| Student | Contribution | Percentage |
|---|---|---|
| Member 1 (Rohan) | Dataset creation, domain task definition, evaluation queries | 35% |
| Member 2 (Blake) | QLoRA fine-tuning, GEPA optimization, FastAPI/Streamlit app, evaluation | 45% |
| Member 3 (Kenneth) | Report compilation, documentation, demo testing | 20% |

### 3. Individual Submissions
Each team member writes a short report with:
- Description of their contributions
- Contribution percentage
- GitHub commits as evidence
- AI tools used (e.g., Claude Code for development assistance)

### 4. Final Commit and Cleanup
Before submitting:
- Make sure all files are committed and pushed
- Verify the repo is clean: `git status`
- Double-check `SETUP_GUIDE.md` is accurate

---

## Running the Demo

1. `pip install -r requirements.txt`
2. `export HF_TOKEN="hf_..."` (free token from https://huggingface.co/settings/tokens)
3. Terminal 1: `uvicorn backend.app:app --port 3001`
4. Terminal 2: `streamlit run frontend/app.py --server.port 3000`
5. Open http://localhost:3000
6. Use **"Compare All"** mode with sample queries from the sidebar

**Demo tips:**
- Click through different sample queries to show all 5 task types
- Point out the score differences between configs in the sidebar
- Expand "View Full Evaluation Results" to show the comparison tables
- Highlight that QLoRA (95%) outperforms all other configs

---

## File Map

```
Lab_8/
├── backend/app.py                         # FastAPI backend
├── frontend/app.py                        # Streamlit UI
├── training/
│   ├── qlora_finetune.ipynb               # QLoRA training notebook
│   ├── gepa_prompt_optimization.ipynb     # GEPA optimization notebook
│   └── gepa_optimized_prompts.json        # GEPA output
├── evaluation/
│   ├── evaluation_queries.md              # 10 eval queries
│   ├── eval_4_configs.ipynb               # Evaluation notebook
│   ├── eval_results.json                  # Full results + responses
│   └── results.md                         # Results with analysis
├── data/
│   └── instruction_dataset.json           # 50 training examples
├── requirements.txt                       # Python deps
├── SETUP_GUIDE.md                         # Full setup instructions
├── HANDOFF.md                             # This file
└── LAB8_OVERVIEW.md                       # Lab requirements/rubric
```
