"""Minimal Streamlit UI for the deterministic interface agent prototype."""

from __future__ import annotations

import json

import streamlit as st

from agents.interface_agent import classify_goal_deterministic


EXAMPLE_PROMPTS = [
    "Given W0 and x0, what D and xDavg can I get?",
    "I have 5 gallons of 12% ABV wash. What product amount and strength could I get?",
    "I want 1 gallon of product at 60% ABV. What feed do I need?",
    "What is xDavg?",
    "Given W0, x0, D, and xDavg, solve for B.",
    "Are W0, x0, D, xDavg, B, and xB physically valid together?",
]


st.set_page_config(page_title="Batch Distillation Interface Agent Prototype", layout="wide")

st.title("Batch Distillation Interface Agent Prototype")
st.write(
    "This prototype only classifies the user's request into a structured goal schema. "
    "It does not run engineering calculations yet."
)

with st.sidebar:
    st.header("Example prompts")
    for prompt in EXAMPLE_PROMPTS:
        st.code(prompt, language=None)

user_request = st.text_area(
    "User request",
    height=180,
    placeholder="Describe the batch distillation question you want classified...",
)

if st.button("Classify goal", type="primary"):
    if not user_request.strip():
        st.warning("Enter a request first.")
    else:
        classification = classify_goal_deterministic(user_request)
        st.subheader("Classification")
        st.json(json.loads(json.dumps(classification.model_dump())))
