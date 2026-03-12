# Lab 8: Task Breakdown by Team Member

## Deadline: Friday, March 13, 2026 (grace period until Monday, March 15 at noon)

---

## Team Member 1 – Dataset Creation & Evaluation (35%)

### Responsibilities

- **Define the domain task** (Step 1)
  - Identify the specific domain reasoning task for the project
  - Document what the expected model output should look like
  - Write justification for why domain adaptation will improve the system

- **Build the instruction dataset** (Step 2)
  - Create 20–50 instruction/input/output examples in JSON format
  - Source examples from project data, research papers, domain documents, and/or AI-generated content
  - Ensure dataset quality and domain relevance
  - Validate dataset formatting for compatibility with the training pipeline

- **Design and run evaluation** (Step 5)
  - Create at least 10 evaluation queries
  - Run queries against both baseline and adapted systems
  - Measure: accuracy, domain relevance, hallucination rate, response clarity
  - Build the comparison table of baseline vs. adapted results
  - Write up evaluation results and analysis for the group report

### Deliverables

- `instruction_dataset.json` (or `.csv`) committed to the repo
- Evaluation queries and results documentation
- Sections of the group report: domain task definition, dataset description, evaluation results

---

## Team Member 2 – Fine-Tuning / Model Adaptation (35%)

### Responsibilities

- **Implement domain adaptation** (Step 3)
  - Choose adaptation method: LoRA/QLoRA fine-tuning (preferred) or prompt adaptation
  - Set up the training environment (Google Colab, local, etc.)
  - Select and load a base model (Mistral, Llama, Phi, etc.)
  - Apply LoRA/QLoRA fine-tuning using HuggingFace Transformers + PEFT library
  - Train the model on the instruction dataset
  - Save and export the adapted model/adapter weights

- **If using Option B (Prompt Adaptation)**
  - Design structured system prompts with domain-specific instructions
  - Implement chain-of-thought prompting strategies
  - Create prompt templates that encode domain knowledge
  - Demonstrate measurable improvement over baseline prompts

- **Model validation**
  - Verify the adapted model loads and generates responses correctly
  - Test basic inference before integration
  - Document training parameters, hyperparameters, and any issues encountered

### Deliverables

- Fine-tuning / adaptation scripts committed to the repo
- Saved model weights or adapter files (or prompt templates if using Option B)
- Sections of the group report: adaptation method used, implementation details

---

## Team Member 3 – Integration, UI & Demo (30%)

### Responsibilities

- **Integrate adapted model into the project pipeline** (Step 4)
  - Update the FastAPI backend to load and serve the adapted model
  - Connect RAG retrieval pipeline to the domain-adapted model
  - Ensure the full pipeline works: Streamlit UI → FastAPI → RAG → Adapted Model → Response
  - Handle any API/endpoint changes needed for the new model

- **Update the Streamlit demo** (Step 6)
  - Add side-by-side or toggle view showing baseline vs. adapted responses
  - Display improvement in domain reasoning clearly
  - Ensure the demo is polished for hackathon presentation

- **Testing & code quality**
  - End-to-end testing of the integrated system
  - Verify all components work together
  - Code cleanup and documentation
  - Ensure the GitHub repo is well-organized

- **Group report coordination**
  - Compile the final group report (1–2 pages, PDF)
  - Ensure the contribution table is included and totals 100%
  - Include GitHub repository link
  - Coordinate with teammates on report sections

### Deliverables

- Updated Streamlit app and FastAPI backend committed to the repo
- Working demo showing baseline vs. adapted responses
- Final group report (compiled from all members' sections)

---

## Shared Responsibilities (All Members)

- **Individual Submission** – each member writes and submits their own:
  - Description of their contributions
  - Contribution percentage
  - GitHub commits/links as evidence
  - AI tools used and how they assisted development
- Keep GitHub commits clean and attributable
- Communicate progress with the team
- Document any AI tools used during development

---

## Contribution Summary

| Team Member | Focus Area | Percentage |
|---|---|---|
| Member 1 | Dataset creation, domain task definition, evaluation | 35% |
| Member 2 | Fine-tuning / model adaptation implementation | 35% |
| Member 3 | Streamlit/FastAPI integration, demo, report assembly | 30% |
| **Total** | | **100%** |

---

## Suggested Timeline

| Date | Milestone |
|---|---|
| Day 1 | Define domain task, start dataset creation, set up training env |
| Day 2 | Complete dataset, begin fine-tuning / adaptation |
| Day 3 | Finish adaptation, start integration |
| Day 4 | Complete integration, run evaluations, update demo |
| Day 5 (Mar 13) | Finalize report, submit group + individual reports |
| Mar 15 (noon) | Hard deadline if using grace period |
