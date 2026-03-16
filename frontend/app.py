"""
Legal Demand Assistant — Streamlit Frontend
=============================================
Chat interface with 4-configuration toggle and side-by-side comparison.
"""

import json

import requests
import streamlit as st

BACKEND_URL = "http://localhost:3001"

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal Demand Assistant",
    page_icon="balance_scale",
    layout="wide",
)

st.title("Legal Demand Assistant")
st.caption("Domain-adapted Qwen3.5-9B | QLoRA Fine-Tuning + GEPA Prompt Optimization")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    mode = st.radio(
        "Mode",
        ["Single Config", "Compare All"],
        help="Single Config runs one config at a time. Compare All shows all 4 side-by-side.",
    )

    if mode == "Single Config":
        config = st.selectbox(
            "Select Configuration",
            ["baseline", "gepa", "qlora", "gepa_qlora"],
            format_func=lambda x: {
                "baseline": "Baseline (no adaptation)",
                "gepa": "GEPA (optimized prompts)",
                "qlora": "QLoRA (fine-tuned) [precomputed]",
                "gepa_qlora": "GEPA + QLoRA [precomputed]",
            }[x],
        )

    st.divider()
    st.header("Evaluation Scores")

    # Load eval summary
    try:
        summary_resp = requests.get(f"{BACKEND_URL}/eval-summary", timeout=5)
        if summary_resp.status_code == 200:
            summary = summary_resp.json()
            for cfg_key in ["baseline", "gepa", "qlora", "gepa_qlora"]:
                if cfg_key in summary:
                    label = summary[cfg_key]["label"]
                    score = summary[cfg_key]["overall_score"]
                    st.metric(label, f"{score:.0%}")
        else:
            st.warning("Could not load evaluation scores")
    except requests.exceptions.ConnectionError:
        st.error("Backend not running. Start with: `uvicorn backend.app:app --port 3001`")

    st.divider()
    st.markdown(
        "**Live configs** (Baseline, GEPA) call the HuggingFace API.\n\n"
        "**Precomputed configs** (QLoRA, GEPA+QLoRA) use responses "
        "generated on an A100 GPU during evaluation."
    )

    st.divider()
    st.header("Sample Queries")
    sample_queries = [
        "Draft a demand letter for a tenant whose landlord withheld a $2,500 security deposit without providing an itemized statement of damages.",
        "Identify the strongest legal claim where a contractor accepted an $8,000 payment, performed minimal work, and abandoned the job.",
        "Extract the claimant, recipient, damages, deadline, and requested remedy from the following demand letter:\n\nDear Mr. Allen: My client, Sarah Kim, seeks payment of $4,800 for water damage caused to her property on January 8, 2026, when your negligence led to a plumbing overflow from your unit. Please remit payment within 14 days of this letter to avoid further action.",
        "Evaluate whether this demand letter is complete and effective:\n\nYou owe me money. Please fix this immediately.",
        "Suggest appropriate remedies when a contractor took a deposit, abandoned a home repair project, and the homeowner had to hire another contractor for additional cost.",
    ]
    for sq in sample_queries:
        if st.button(sq[:60] + "...", key=sq[:30]):
            st.session_state["query_input"] = sq


# ── Helper Functions ─────────────────────────────────────────────────────────

def call_backend(query: str, config: str) -> dict:
    """Call the backend API."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"query": query, "config": config},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"response": f"Error: {resp.text}", "config_label": config, "is_live": False, "task_type": "unknown"}
    except requests.exceptions.ConnectionError:
        return {"response": "Error: Backend not running", "config_label": config, "is_live": False, "task_type": "unknown"}


# ── Main Interface ───────────────────────────────────────────────────────────

query = st.text_area(
    "Enter your legal query:",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g., Draft a demand letter for a freelancer who completed work but was never paid...",
)

if st.button("Submit", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        if mode == "Single Config":
            # ── Single Config Mode ──
            with st.spinner(f"Running {config}..."):
                result = call_backend(query, config)

            st.subheader(f"Response — {result.get('config_label', config)}")

            # Show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                source = "Live (HF API)" if result.get("is_live") else "Precomputed (A100)"
                st.caption(f"Source: {source}")
            with col2:
                st.caption(f"Task: {result.get('task_type', 'unknown')}")
            with col3:
                if result.get("score") is not None:
                    st.caption(f"Score: {result['score']:.0%}")

            st.markdown(result["response"])

        else:
            # ── Compare All Mode ──
            configs = ["baseline", "gepa", "qlora", "gepa_qlora"]
            results = {}

            with st.spinner("Running all 4 configurations..."):
                for cfg in configs:
                    results[cfg] = call_backend(query, cfg)

            # Display in 2x2 grid
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            grid = [
                (row1_col1, "baseline"),
                (row1_col2, "gepa"),
                (row2_col1, "qlora"),
                (row2_col2, "gepa_qlora"),
            ]

            for col, cfg in grid:
                with col:
                    r = results[cfg]
                    label = r.get("config_label", cfg)
                    source = "Live" if r.get("is_live") else "Precomputed"
                    score_str = f" | Score: {r['score']:.0%}" if r.get("score") is not None else ""

                    st.subheader(label)
                    st.caption(f"{source}{score_str}")
                    st.markdown(r["response"])


# ── Evaluation Results Tab ───────────────────────────────────────────────────

st.divider()

with st.expander("View Full Evaluation Results", expanded=False):
    try:
        eval_resp = requests.get(f"{BACKEND_URL}/eval-results", timeout=5)
        if eval_resp.status_code == 200:
            eval_data = eval_resp.json()
            summary = eval_data.get("summary", {})

            # Comparison table
            st.subheader("Per-Query Scores")
            table_data = []
            queries = [f"Q{i}" for i in range(1, 11)]
            for q in queries:
                row = {"Query": q}
                for cfg_key in ["baseline", "gepa", "qlora", "gepa_qlora"]:
                    if cfg_key in summary:
                        score = summary[cfg_key]["per_query"].get(q, 0)
                        row[summary[cfg_key]["label"]] = f"{score:.0%}"
                table_data.append(row)

            # Add average row
            avg_row = {"Query": "Average"}
            for cfg_key in ["baseline", "gepa", "qlora", "gepa_qlora"]:
                if cfg_key in summary:
                    avg_row[summary[cfg_key]["label"]] = f"{summary[cfg_key]['overall_score']:.0%}"
            table_data.append(avg_row)

            st.table(table_data)

            # Per-task breakdown
            st.subheader("Per-Task Breakdown")
            task_types = ["draft_demand_letter", "identify_claim", "extract_elements", "evaluate_letter", "recommend_remedy"]
            task_data = []
            for task in task_types:
                row = {"Task": task}
                for cfg_key in ["baseline", "gepa", "qlora", "gepa_qlora"]:
                    if cfg_key in summary:
                        score = summary[cfg_key]["per_task"].get(task, 0)
                        row[summary[cfg_key]["label"]] = f"{score:.0%}"
                task_data.append(row)
            st.table(task_data)

    except requests.exceptions.ConnectionError:
        st.error("Backend not running.")
