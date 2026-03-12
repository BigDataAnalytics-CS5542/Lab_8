# Lab 8: Domain Adaptation Approaches

## Background: GEPA-lite

Our team has an existing project, **GEPA-lite** (Genetic-Pareto), a prompt/text optimization framework that uses LLM-based reflection and evolutionary search. GEPA automates prompt adaptation by:

1. Running candidate prompts against evaluation examples
2. Using LLM reflection on execution traces to diagnose failures
3. Mutating candidates to improve performance
4. Tracking Pareto-optimal solutions across evaluation dimensions

GEPA includes a **Generic RAG Adapter** that can optimize query reformulation, context synthesis, answer generation, and document reranking — directly applicable to our project pipeline.

---

## Approach 1: GEPA Only (Prompt Adaptation)

Use GEPA to evolve the system prompt and RAG prompt for our domain task.

### Workflow

```
Build instruction dataset (20–50 examples)
  ↓
Configure GEPA with Generic RAG Adapter
  ↓
Run evolutionary optimization on prompts
  ↓
Select best Pareto-optimal prompt candidate
  ↓
Integrate optimized prompt into pipeline
```

### Pros

- Leverages our existing GEPA-lite project directly
- No GPU or fine-tuning infrastructure needed
- Novel approach — demonstrates understanding of automated prompt optimization
- Fast iteration (100–500 metric calls vs thousands for RL)

### Cons

- Falls under "Option B" (prompt adaptation) — the lab prefers fine-tuning (Option A)
- May score lower on the "Model adaptation implementation" rubric (30 pts)

---

## Approach 2: LoRA / QLoRA Fine-Tuning Only

Fine-tune a small open-source model on our instruction dataset using parameter-efficient methods.

### Workflow

```
Build instruction dataset (20–50 examples)
  ↓
Select base model (Phi-3 Mini, Mistral 7B, or Llama 3 8B)
  ↓
Apply QLoRA fine-tuning via HuggingFace PEFT
  ↓
Train on Google Colab
  ↓
Save adapter weights
  ↓
Integrate fine-tuned model into pipeline
```

### Model Options

| Model | Size | VRAM Needed | Colab Tier |
|---|---|---|---|
| Phi-3 Mini | 3.8B | ~8 GB (QLoRA) | Free (T4) |
| Mistral 7B | 7B | ~12 GB (QLoRA) | Free (T4) |
| Llama 3 8B | 8B | ~12 GB (QLoRA) | Free (T4) |
| Phi-3 Medium | 14B | ~16 GB (QLoRA) | Pro (A100) |

### Pros

- Directly satisfies the lab's preferred method (Option A)
- Strong score potential on "Model adaptation implementation" (30 pts)
- Straightforward implementation with well-documented libraries

### Cons

- Does not leverage our existing GEPA-lite project
- Requires GPU compute (Colab free tier should suffice for smaller models)
- 20–50 examples is small for fine-tuning — results may be modest

---

## Approach 3: GEPA + LoRA Comparison (Recommended)

Combine both approaches and present a comparative evaluation. This is the strongest option for the lab.

### Workflow

```
Build instruction dataset (20–50 examples)
  ↓
┌──────────────────────────────┬──────────────────────────────┐
│  Track A: QLoRA Fine-Tuning  │  Track B: GEPA Optimization  │
│                              │                              │
│  Select base model           │  Configure GEPA RAG Adapter  │
│  Apply QLoRA via PEFT        │  Run evolutionary search     │
│  Train on Colab              │  Select Pareto-optimal       │
│  Save adapter weights        │  prompt candidates           │
└──────────────┬───────────────┴──────────────┬───────────────┘
               ↓                              ↓
         Integrate both into pipeline (toggle between them)
               ↓
         Evaluate all configurations
```

### Evaluation Matrix

| Configuration | Description |
|---|---|
| Baseline | Original system, no adaptation |
| GEPA-optimized | Evolved prompts via GEPA reflective mutation |
| QLoRA fine-tuned | Parameter-efficient fine-tuned open-source model |
| GEPA + QLoRA | GEPA-optimized prompts feeding into the fine-tuned model |

### System Architecture

```
Streamlit UI (with comparison toggle)
  ↓
FastAPI Backend
  ↓
RAG Retrieval
  ↓
┌─────────────────────────────────┐
│  Model Selection (toggle):      │
│   • Baseline                    │
│   • GEPA-optimized prompts      │
│   • QLoRA fine-tuned model      │
│   • GEPA + QLoRA combined       │
└─────────────────────────────────┘
  ↓
Response (with side-by-side comparison view)
```

### Evaluation Metrics (10+ queries)

| Metric | Description |
|---|---|
| Accuracy | Correctness of domain-specific responses |
| Domain relevance | Quality and depth of reasoning |
| Hallucination rate | Frequency of incorrect information |
| Response clarity | Explanation quality and structure |

### Pros

- Satisfies both Option A (fine-tuning) and Option B (prompt adaptation)
- Richer evaluation section — comparative analysis across multiple methods
- Demonstrates deep understanding of domain adaptation techniques
- Directly ties into our existing GEPA-lite project
- Strongest potential score across all rubric categories
- The comparison itself is a valuable contribution to the report

### Cons

- More work to implement (mitigated by splitting across 3 team members)
- Need to manage two adaptation pipelines

---

## Recommendation

**Go with Approach 3.** It maximizes our grade potential across all rubric categories:

| Rubric Category (100 pts) | How Approach 3 Scores |
|---|---|
| Instruction dataset quality (25) | Same high-quality dataset used by both methods |
| Model adaptation implementation (30) | QLoRA satisfies Option A; GEPA adds depth |
| Integration with project pipeline (20) | Toggle UI showing all configurations |
| Evaluation and analysis (15) | 4-way comparison is richer than 2-way |
| Code quality and documentation (10) | Well-organized repo with clear separation |

### Task Mapping to Team Members

| Member | Approach 3 Responsibilities |
|---|---|
| Member 1 (35%) | Build instruction dataset, design evaluation queries, run evaluations across all 4 configurations |
| Member 2 (35%) | Implement QLoRA fine-tuning + configure GEPA optimization |
| Member 3 (30%) | Integrate both into Streamlit/FastAPI with toggle UI, assemble report |
