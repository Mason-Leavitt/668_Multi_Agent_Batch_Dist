import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agents.design_advisor_helpers import format_scenario_row
from agents.experiment_intents import (
    ExperimentFollowupConfidence,
    ExperimentFollowupIntent,
    build_experiment_intent,
    unknown_experiment_intent,
)

load_dotenv()


class LLMExperimentFollowupIntent(BaseModel):
    is_experiment_followup: bool
    intent: ExperimentFollowupIntent
    option_index: int | None = None
    target_variable: str | None = None
    value: float | None = None
    relative_choice: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    confidence: ExperimentFollowupConfidence = "interpreted"

def is_plausible_experiment_followup(user_message: str) -> bool:
    message = (user_message or "").strip().lower()
    if not message:
        return False

    followup_phrases = (
        "option",
        "try",
        "compare",
        "vary",
        "higher",
        "lower",
        "show",
        "use",
        "choose",
        "pick",
        "what if",
        "suppose",
        "explain",
        "end",
        "done",
        "go with",
        "cases",
        "feed composition",
        "starting concentration",
        "stopping composition",
        "x0",
        "xb",
        "xdavg",
    )
    if any(phrase in message for phrase in followup_phrases):
        return True

    return False


def _summarize_experiment_context(
    active_experiment: dict | None,
    experiment_results: list[dict] | None,
    experiment_sampled_variable: str | None,
    experiment_knowns: dict | None,
) -> str:
    rows = experiment_results or []
    scenario_lines = []
    for index, row in enumerate(rows[:5], start=1):
        scenario_lines.append(f"option {index}: {format_scenario_row(row)}")

    known_lines = []
    for key, value in (experiment_knowns or {}).items():
        if isinstance(value, float):
            known_lines.append(f"- {key} = {value:.6f}")
        else:
            known_lines.append(f"- {key} = {value}")

    return (
        f"Active sampled variable: {experiment_sampled_variable or active_experiment.get('sampled_variable') if active_experiment else experiment_sampled_variable}\n"
        f"Knowns:\n" + ("\n".join(known_lines) if known_lines else "- none") + "\n"
        f"Stored scenario rows: {len(rows)}\n"
        f"Scenario options:\n" + ("\n".join(scenario_lines) if scenario_lines else "- none")
    )


def interpret_experiment_followup_with_llm(
    user_message: str,
    active_experiment: dict,
    experiment_results: list[dict] | None,
    experiment_sampled_variable: str | None,
    experiment_knowns: dict | None,
) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        return unknown_experiment_intent()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        LLMExperimentFollowupIntent,
        method="function_calling",
    )

    prompt = f"""
You are interpreting a follow-up message within an active experiment for a batch-distillation assistant.
Your job is only to map the user's message into a structured experiment-followup intent.
Do not perform calculations. Do not invent values. Be conservative.

Allowed intents:
- select_option
- try_custom_value
- shift_samples_higher
- shift_samples_lower
- switch_sampling_axis
- end_experiment
- explain_option
- unknown

Variable aliases:
- feed composition / starting concentration / initial ethanol concentration -> x0
- stopping composition / final still composition / bottoms composition -> xB
- average distillate composition -> xDavg_target when discussing target scenarios
- option / row / case -> scenario option

Rules:
- If the user is ambiguous, set needs_clarification=true and provide a short clarification_question.
- If you cannot confidently map the message, use intent="unknown", is_experiment_followup=false, confidence="low".
- For "middle option", you may use relative_choice="middle" if no exact option index is stated.
- For "first" or "last", use relative_choice accordingly.
- For "explain option 2", use intent="explain_option".
- Confidence for this interpreter should be "interpreted" unless truly unknown.

Current experiment context:
{_summarize_experiment_context(active_experiment, experiment_results, experiment_sampled_variable, experiment_knowns)}

User message:
{user_message}
"""

    try:
        interpreted = structured_llm.invoke(prompt)
        data = interpreted.model_dump()
        return build_experiment_intent(
            data["intent"],
            is_experiment_followup=data["is_experiment_followup"],
            option_index=data.get("option_index"),
            target_variable=data.get("target_variable"),
            value=data.get("value"),
            relative_choice=data.get("relative_choice"),
            needs_clarification=data.get("needs_clarification", False),
            clarification_question=data.get("clarification_question"),
            confidence=data.get("confidence", "interpreted"),
        )
    except Exception:
        return unknown_experiment_intent()
