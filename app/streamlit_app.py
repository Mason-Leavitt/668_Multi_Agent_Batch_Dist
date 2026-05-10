"""Minimal Streamlit UI for the interface agent classifier prototype."""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from agents.interface_agent import classify_goal


st.set_page_config(page_title="Batch Distillation Interface Agent Prototype", layout="wide")

st.title("Batch Distillation Interface Agent Prototype")
st.write(
    "This prototype only classifies the user's request into a structured goal schema. "
    "It does not run engineering calculations yet."
)

with st.sidebar:
    st.header("Classifier settings")
    classifier_mode = st.radio(
        "Classification mode",
        options=["LLM classifier", "Deterministic fallback"],
        index=0,
    )
    model_name = st.text_input("Model name", value="gpt-4o-mini")
    st.divider()
    st.header("Live assessment")

if "submitted_request" not in st.session_state:
    st.session_state.submitted_request = ""
if "last_warning" not in st.session_state:
    st.session_state.last_warning = None

st.write("Enter a request below and press Enter to submit it.")

submitted_request = st.chat_input("Describe the batch distillation question you want classified...")
if submitted_request:
    st.session_state.submitted_request = submitted_request
    st.session_state.last_warning = None

user_request = st.session_state.submitted_request
classification = None

if user_request.strip():
    use_llm = classifier_mode == "LLM classifier"
    classification = classify_goal(
        user_request,
        use_llm=use_llm,
        model_name=model_name.strip() or "gpt-4o-mini",
    )
    if use_llm and classification.reasoning_summary.startswith("Fallback used after LLM error:"):
        st.session_state.last_warning = (
            "LLM classification failed, so the deterministic fallback was used instead."
        )

with st.sidebar:
    st.subheader("Current goal")
    if classification is None:
        st.info("No goal assessed yet.")
    else:
        st.code(classification.goal, language=None)
        st.caption(classification.user_facing_summary)

    st.subheader("Variable assignments")
    if classification is None or not classification.variable_assignments:
        st.info("No variable assignments detected yet.")
    else:
        st.json(classification.variable_assignments)

if classification is None:
    st.info("Enter a request to see the live classification.")
else:
    if st.session_state.last_warning:
        st.warning(st.session_state.last_warning)

    with st.chat_message("user"):
        st.write(user_request)

    st.subheader("Classification")
    st.json(classification.model_dump())
