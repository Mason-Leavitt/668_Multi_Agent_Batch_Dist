"""Minimal Streamlit UI for the interface agent classifier prototype."""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from agents.interface_agent import analyze_message_features, classify_goal


st.set_page_config(page_title="Batch Distillation Interface Agent Prototype", layout="wide")

st.title("Batch Distillation Interface Agent Prototype")
st.write(
    "This prototype only classifies the user's request into a structured goal schema. "
    "It does not run engineering calculations yet."
)

with st.sidebar:
    st.header("Classifier settings")
    st.caption("LLM structured classifier")
    model_name = st.text_input("Model name", value="gpt-4o-mini")
    show_hints = st.checkbox("Show lightweight detected hints", value=False)
    st.divider()
    st.header("Live assessment")

if "submitted_request" not in st.session_state:
    st.session_state.submitted_request = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = None

st.write("Enter a request below and press Enter to submit it.")

submitted_request = st.chat_input("Describe the batch distillation question you want classified...")
if submitted_request:
    st.session_state.submitted_request = submitted_request
    st.session_state.last_error = None

user_request = st.session_state.submitted_request
classification = None
feature_hints = analyze_message_features(user_request) if user_request.strip() else None

if user_request.strip():
    try:
        classification = classify_goal(
            user_request,
            model_name=model_name.strip() or "gpt-4o-mini",
        )
    except Exception as exc:
        st.session_state.last_error = str(exc)

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

    if show_hints:
        st.subheader("Detected hints")
        if feature_hints is None:
            st.info("No hints detected yet.")
        else:
            st.json(feature_hints)

if classification is None and not st.session_state.last_error:
    st.info("Enter a request to see the live classification.")
elif st.session_state.last_error:
    st.error(
        "LLM classification failed. No deterministic fallback was used because fallback classifications may be misleading."
    )
    with st.expander("Debug error details"):
        st.code(st.session_state.last_error)
else:
    with st.chat_message("user"):
        st.write(user_request)

    st.subheader("Classification")
    st.json(classification.model_dump())
