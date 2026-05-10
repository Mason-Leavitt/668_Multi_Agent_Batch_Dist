"""LLM-based semantic router plus lightweight context extraction helpers."""

from __future__ import annotations

import json
import re

from app.ui_metadata import get_workflow_reference

from .router_schemas import ConversationRoute


def build_router_workflow_summary_section() -> str:
    lines = ["Implemented executable workflows:"]
    for workflow in get_workflow_reference():
        workflow_id = str(workflow["workflow_id"])
        display_name = str(workflow["display_name"])
        description = str(workflow["description"])
        required_inputs = ", ".join(str(item) for item in workflow.get("required_inputs", []))
        outputs = ", ".join(str(item) for item in workflow.get("outputs", []))
        lines.append("")
        lines.append(f"{workflow_id} ({display_name}):")
        lines.append(description)
        if required_inputs:
            lines.append(f"- Typical required inputs: {required_inputs}.")
        if outputs:
            lines.append(f"- Typical outputs: {outputs}.")
    return "\n".join(lines)


WORKFLOW_SUMMARY_SECTION = build_router_workflow_summary_section()


ROUTER_SYSTEM_PROMPT = f"""
You are the semantic conversation router for a batch distillation assistant.

Your only job is to route the user's conversational message. Do not perform
calculations. Do not invent numeric results. Do not replace the separate
GoalClassification step.

{WORKFLOW_SUMMARY_SECTION}

Route meanings:

confirm_pending_workflow:
Use when the app is waiting for confirmation and the user says yes, yeah,
proceed, run it, go ahead, execute, do it, calculate it, or equivalent.

cancel_pending_workflow:
Use when the app is waiting for confirmation and the user says no, cancel,
stop, not yet, wait, revise, change it, or equivalent.

new_workflow_request:
Use when the user is asking to perform a new batch distillation calculation,
sweep, design, or check.

Examples:
- "i want an output of 10L at 45% abv. what inputs can i use to achieve that?"
- "given a product amount of 10L at 45% abv, what feed conditions might work?"
- "i have 100 L at 10% ABV. what product outcomes can i get?"
- "given W0 = 100 mol and x0 = 0.05, what D and xDavg can I get?"
- "I want to produce 2 gallons at 60% ABV. What feed do I need?"
- "I started with 100 L at 10% ABV and collected 10 L at 45% ABV. What is left in the still?"
- "Use the mole balance to solve for B and xB."
- "Given D = 20 mol, xDavg = 0.4, B = 80 mol, and xB = 0.02, what W0 and x0 did I start with?"
- "Given W0 = 100 mol, x0 = 0.05, and xB = 0.01, calculate D and xDavg."
- "Given W0 = 100 mol, x0 = 0.05, and D = 20 mol, calculate xDavg."
- "If I want 10 L at 45% ABV and my feed is 10% ABV, how much feed do I need?"
- "If I want 10 L at 45% ABV and stop the boiler at 2% ABV, what feed do I need?"

workflow_clarification:
Use when the user is continuing an incomplete workflow setup or clarifying an
intended workflow.

Examples:
- The user previously said they want feed options from desired outputs, then says "10 L at 45% ABV."
- The user says "I mean product to feed."
- The user says "use the output target I gave earlier."

result_question:
Use when the user asks a targeted question about the most recent result table or plot.

Examples:
- "based on these results, how much distillate can I get at 40% ABV?"
- "which row gives the most product?"
- "what feed volume corresponds to 45% feed ABV?"
- "what does the table say at xDavg around 50%?"
- "based on the results, if i used an input composition of 10% abv, what would the feed volume be?"
- "and what is the feed volume for that row?"
- "so given these results, if i used a feed composition of 10% abv, how much feed would i need?"
- "given these results if the input composition is 10% abv, what amount of feed would i need?"

result_explanation_request:
Use when the user asks to explain or interpret the most recent results.

Examples:
- "explain"
- "explain these results"
- "what does this plot mean?"
- "summarize the result"

general_conversation:
Use for ordinary discussion, corrections, conceptual questions, app
capabilities, and non-actionable messages.

Examples:
- "that wording is wrong"
- "D is mixture amount, not pure ethanol"
- "what can this app do?"
- "what is xDavg?"
- "why do you use mole fractions?"
- "thanks"

Important routing rule:
General conversation is the fallback, but do not route actionable workflow
requests to general_conversation.

Important workflow distinction:
- Known feed/input + asks what product/output can be achieved = feed_to_product_sweep.
- Desired product/output + asks what feed/input can achieve it = product_to_feed_sweep.

Important user-unit rule:
Volume and ABV are normal, sufficient user-facing inputs. Do not require moles
or mole fractions when volume and ABV are provided.

For:
"i want an output of 10L at 45% abv. what inputs can i use to achieve that?"
route = new_workflow_request
likely_goal = product_to_feed_sweep
should_call_goal_classifier = true
confidence should be high or moderate-high

For:
"given a product amount of 10L at 45% abv, what feed conditions might work?"
route = new_workflow_request
likely_goal = product_to_feed_sweep
should_call_goal_classifier = true

For:
"i have an input volume of 100L at 10% abv. What product amount and compositions could i achieve?"
route = new_workflow_request
likely_goal = feed_to_product_sweep
should_call_goal_classifier = true

For:
"I started with 100 L at 10% ABV and collected 10 L at 45% ABV. What is left in the still?"
route = new_workflow_request
likely_goal = solve_mole_balance
should_call_goal_classifier = true

For:
"Given W0 = 100 mol, x0 = 0.05, and xB = 0.01, calculate D and xDavg."
route = new_workflow_request
likely_goal = solve_rayleigh_batch_variables
should_call_goal_classifier = true

For:
"it is not 382 liters of ethanol, it is 382 liters of mixture"
route = general_conversation
should_call_goal_classifier = false

For:
"explain these results"
route = result_explanation_request if last_execution_result exists

For:
"based on the results, if i used an input composition of 10% abv, what would the feed volume be?"
route = result_question if last_execution_result exists

For:
"and what is the feed volume for that row?"
route = result_question if there is a previously matched result row

For:
"so given these results, if i used a feed composition of 10% abv, how much feed would i need?"
route = result_question if last_execution_result exists

For:
"given these results if the input composition is 10% abv, what amount of feed would i need?"
route = result_question if last_execution_result exists

For:
"yes"
route = confirm_pending_workflow only if awaiting_confirmation is true.
Otherwise general_conversation.

Return only the structured ConversationRoute object.
""".strip()


CONFIRM_TERMS = {
    "yes",
    "y",
    "yeah",
    "yep",
    "run it",
    "go ahead",
    "proceed",
    "execute",
    "do it",
    "calculate it",
}

CANCEL_TERMS = {
    "no",
    "cancel",
    "stop",
    "not yet",
    "change",
    "revise",
    "wait",
}


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def extract_volume_abv_pair(text: str) -> dict[str, str] | None:
    normalized = _normalize_text(text)
    volume_match = re.search(
        r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>l|liter|liters|litre|litres|ml|gallon|gallons|gal)\b",
        normalized,
    )
    abv_match = re.search(
        r"\b(?P<value>\d+(?:\.\d+)?)\s*(?:%\s*(?:abv)?|percent(?:\s+abv)?|abv)\b",
        normalized,
    )
    if not volume_match or not abv_match:
        return None

    volume_unit = volume_match.group("unit")
    if volume_unit == "l":
        volume_unit = "L"
    elif volume_unit == "ml":
        volume_unit = "mL"

    return {
        "volume": f"{volume_match.group('value')} {volume_unit}",
        "abv": f"{abv_match.group('value')}% ABV",
    }


def infer_target_role(text: str) -> str | None:
    normalized = _normalize_text(text)
    product_terms = ["product", "output", "distillate", "collect", "produce", "target", "achieve"]
    feed_terms = ["feed", "input", "wash", "starting", "charge"]

    has_product_terms = any(term in normalized for term in product_terms)
    has_feed_terms = any(term in normalized for term in feed_terms)

    if has_product_terms and ("what input" in normalized or "what feed" in normalized or "starting conditions" in normalized):
        return "product"
    if has_feed_terms and ("what output" in normalized or "what product" in normalized or "what can i get" in normalized):
        return "feed"
    if has_product_terms and not has_feed_terms:
        return "product"
    if has_feed_terms and not has_product_terms:
        return "feed"
    return None


def infer_likely_goal(text: str) -> str | None:
    normalized = _normalize_text(text)
    if extract_volume_abv_pair(normalized):
        target_role = infer_target_role(normalized)
        if target_role == "product":
            return "product_to_feed_sweep"
        if target_role == "feed":
            return "feed_to_product_sweep"

    if "product to feed" in normalized:
        return "product_to_feed_sweep"
    if "feed to product" in normalized:
        return "feed_to_product_sweep"
    return None


def safe_confirmation_fallback(user_message: str, awaiting_confirmation: bool) -> str | None:
    normalized = _normalize_text(user_message)
    if not awaiting_confirmation:
        return None
    if normalized in CONFIRM_TERMS:
        return "confirm_pending_workflow"
    if normalized in CANCEL_TERMS:
        return "cancel_pending_workflow"
    return None


def route_conversation_message(
    user_message: str,
    conversation_context: dict,
    model_name: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> ConversationRoute:
    """Route a conversational message semantically with structured output."""

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(
        ConversationRoute,
        method="function_calling",
        include_raw=False,
    )

    result = structured_llm.invoke(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            (
                "human",
                "User message:\n"
                f"{user_message}\n\n"
                "Compact conversation context:\n"
                f"{json.dumps(conversation_context, indent=2)}\n\n"
                "Route the message semantically. Use previous context when needed, but do not "
                "pretend the user asked to execute something if the context does not support it.",
            ),
        ]
    )
    return result
