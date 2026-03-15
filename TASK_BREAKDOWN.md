# Lab 8: Task Breakdown by Team Member

## Domain: Legal Demand

The project domain is **legal demand** — the system will assist with drafting, analyzing, and evaluating demand letters and related legal claims. The RAG pipeline will retrieve relevant statutes, demand letter templates, and case summaries to support domain-adapted responses.

## Deadline: Sunday, March 15, 2026 at midnight

---

## Team Member 1 – Dataset Creation & Evaluation (35%)

### Responsibilities

- **Define the domain task** (Step 1)
  - Domain task: Analyzing legal scenarios and drafting/evaluating demand letters
  - Expected output: Properly structured demand letters, legal claim identification, element extraction, and remedy recommendations
  - Justification: Baseline LLMs produce generic legal language — domain adaptation should improve specificity around statutes, required elements, proper tone, and actionable remedies

- **Build the instruction dataset** (Step 2)
  - Create 20–50 instruction/input/output examples in JSON format covering:
    - Drafting demand letters from fact patterns
    - Identifying legal claim types from scenarios
    - Extracting key elements (parties, damages, deadlines, remedies) from demand letters
    - Evaluating whether a demand letter meets formal requirements
    - Suggesting appropriate remedies based on claim type
  - Source examples from legal templates, sample demand letters, statutes, and AI-generated content
  - Ensure dataset quality and domain relevance
  - Validate dataset formatting for compatibility with the training pipeline

- **Design and run evaluation** (Step 5)
  - Create at least 10 evaluation queries
  - Run queries against all 4 configurations: Baseline, GEPA-optimized, QLoRA fine-tuned, GEPA + QLoRA combined
  - Measure: accuracy, domain relevance, hallucination rate, response clarity
  - Build the 4-way comparison table across all configurations
  - Write up evaluation results and comparative analysis for the group report

### Deliverables

- `instruction_dataset.json` (or `.csv`) committed to the repo
- Evaluation queries and 4-way comparison results documentation
- Sections of the group report: domain task definition, dataset description, comparative evaluation results

---

## Team Member 2 (Blake) – Model Adaptation, Integration & Demo (45%)

### Responsibilities

- **Track A: QLoRA Fine-Tuning** (Step 3, Option A)
  - Set up the training environment (Google Colab)
  - Load Qwen3.5-9B as the base model on Colab Pro (A100)
  - Apply QLoRA fine-tuning using HuggingFace Transformers + PEFT library
  - Train the model on the instruction dataset
  - Save and export adapter weights

- **Track B: GEPA Prompt Optimization** (Step 3, Option B)
  - Configure GEPA-lite with the Generic RAG Adapter for legal demand tasks
  - Optimize prompts for legal reasoning: claim identification, demand letter structure, statute citation
  - Run evolutionary optimization on system/RAG prompts
  - Select Pareto-optimal prompt candidates
  - Document GEPA configuration and optimization results

- **Model/prompt validation**
  - Verify the QLoRA-adapted model loads and generates responses correctly
  - Verify GEPA-optimized prompts produce improved responses
  - Test the combined configuration (GEPA prompts + QLoRA model)
  - Document training parameters, hyperparameters, and GEPA settings

- **Integrate both adaptation tracks into the project pipeline** (Step 4)
  - Update the FastAPI backend to support all 4 configurations: Baseline, GEPA-optimized, QLoRA fine-tuned, GEPA + QLoRA combined
  - Add model/prompt selection logic (toggle between configurations)
  - Connect RAG retrieval pipeline to each configuration
  - Ensure the full pipeline works: Streamlit UI → FastAPI → RAG → Selected Configuration → Response

- **Update the Streamlit demo** (Step 6)
  - Add comparison toggle/dropdown to switch between all 4 configurations
  - Add side-by-side comparison view showing responses from different configurations
  - Display improvement metrics and differences clearly
  - Ensure the demo is polished for hackathon presentation

- **Testing & code quality**
  - End-to-end testing of all 4 configurations
  - Verify all components work together
  - Code cleanup and documentation
  - Ensure the GitHub repo is well-organized

### Deliverables

- QLoRA fine-tuning scripts committed to the repo
- Saved adapter weights
- GEPA configuration and optimized prompt artifacts
- Updated Streamlit app and FastAPI backend with 4-configuration toggle
- Working demo showing comparison across all configurations

---

## Team Member 3 – Report & Documentation (20%)

### Responsibilities

- **Group report coordination**
  - Compile the final group report (1–2 pages, PDF)
  - Write up adaptation methods, integration details, and comparative methodology based on Member 2's implementation
  - Ensure the contribution table is included and totals 100%
  - Include GitHub repository link
  - Coordinate with teammates on report sections

- **Documentation**
  - Ensure all code and configuration is documented
  - Review repo organization and clean up as needed

### Deliverables

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
| Member 2 (Blake) | QLoRA fine-tuning, GEPA optimization, FastAPI/Streamlit integration, demo | 45% |
| Member 3 | Report compilation and documentation | 20% |
| **Total** | | **100%** |

---

## Suggested Timeline

| Date | Milestone |
|---|---|
| **Tonight (Sat Mar 14)** | Define domain task, build instruction dataset, set up QLoRA training env + GEPA config, begin both adaptation tracks |
| **Sun Mar 15 – Morning** | Finish QLoRA fine-tuning + GEPA optimization, start integration with 4-config toggle |
| **Sun Mar 15 – Afternoon** | Complete integration, run 4-way evaluations, update Streamlit demo |
| **Sun Mar 15 – Evening** | Finalize comparative analysis, compile report, submit group + individual reports |
| **Sun Mar 15 at midnight** | **Hard deadline** |
