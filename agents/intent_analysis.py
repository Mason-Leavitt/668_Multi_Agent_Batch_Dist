"""Soft intent and shared capability analysis for incomplete workflow requests."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.ui_metadata import get_core_batch_variable_keys, get_missing_value_display_name

from .schemas import GoalClassification, ScalarValue
from .workflow_schemas import WorkflowPlan


class CapabilityCalculationPath(BaseModel):
    """A candidate calculation path derived from the currently known quantities."""

    calculation_id: str
    display_name: str
    known_inputs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    can_execute: bool = False
    explanation: str
    suggested_next_question: str | None = None


class CapabilityAnalysis(BaseModel):
    """Shared capability reasoning for direct-solve and balance-style workflows."""

    known_quantities: dict[str, ScalarValue] = Field(default_factory=dict)
    missing_quantities: list[str] = Field(default_factory=list)
    possible_calculations: list[CapabilityCalculationPath] = Field(default_factory=list)
    nearly_possible_calculations: list[CapabilityCalculationPath] = Field(default_factory=list)
    executable_calculations: list[CapabilityCalculationPath] = Field(default_factory=list)
    blocking_missing_values: list[str] = Field(default_factory=list)
    suggested_next_questions: list[str] = Field(default_factory=list)
    preferred_next_question: str | None = None
    reasoning_summary: str
    can_calculate_now: bool = False


class IntentAnalysis(BaseModel):
    """Soft interpretation of a user request before strict execution planning."""

    raw_user_message: str
    domain_relevant: bool
    likely_task_family: str | None = None
    candidate_goals: list[str] = Field(default_factory=list)
    known_quantities: dict[str, ScalarValue] = Field(default_factory=dict)
    known_quantities_needing_normalization: dict[str, ScalarValue] = Field(default_factory=dict)
    requested_outputs: list[str] = Field(default_factory=list)
    possible_calculations: list[str] = Field(default_factory=list)
    missing_but_needed: list[str] = Field(default_factory=list)
    can_calculate_now: bool = False
    best_next_question: str | None = None
    explanation_hint: str | None = None
    confidence: float | None = None
    capability_analysis: CapabilityAnalysis | None = None


_DOMAIN_TERMS = (
    "w0",
    "x0",
    "xb",
    "xdavg",
    "abv",
    "rayleigh",
    "distill",
    "ethanol",
    "feed",
    "bottoms",
    "still",
    "boiler",
    "product",
    "mole balance",
    "batch",
)

_INCOMPLETE_FOLLOWUP_TERMS = (
    "what additional values do you need",
    "what else do you need",
    "what is missing",
    "what values are missing",
    "why can't you run it",
    "why cant you run it",
    "what would make it solvable",
    "what case is supported",
    "what can you calculate from that",
    "what can you calculate from these",
    "what more do you need",
    "what are my options",
    "can you solve it now",
    "can you sweep it instead",
)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _looks_user_friendly_value(value: ScalarValue) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return (
        "%" in lowered
        or "abv" in lowered
        or "gal" in lowered
        or "gallon" in lowered
        or "liter" in lowered
        or "litre" in lowered
        or " ml" in lowered
        or lowered.endswith("l")
    )


def _is_domain_relevant(user_message: str, classification: GoalClassification | None) -> bool:
    normalized = _normalize_text(user_message)
    if classification is not None and classification.goal != "unsupported_or_unclear":
        return True
    if any(term in normalized for term in _DOMAIN_TERMS):
        return True
    return bool(re.search(r"\b(?:w0|x0|xdavg|xb|abv)\b", normalized))


def _infer_likely_task_family(classification: GoalClassification | None) -> str | None:
    if classification is None:
        return None
    mapping = {
        "feed_to_product_sweep": "feed_to_product",
        "product_to_feed_sweep": "product_to_feed",
        "solve_rayleigh_batch_variables": "direct_rayleigh",
        "solve_mole_balance": "mole_balance",
        "consistency_check": "consistency_check",
        "explain_variable_or_workflow": "explanation",
    }
    return mapping.get(classification.goal)


def _collect_known_quantities(classification: GoalClassification | None) -> tuple[dict[str, ScalarValue], dict[str, ScalarValue]]:
    if classification is None:
        return {}, {}

    known: dict[str, ScalarValue] = {}
    needs_normalization: dict[str, ScalarValue] = {}
    for key, value in classification.variable_assignments.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        known[key] = value
        if _looks_user_friendly_value(value):
            needs_normalization[key] = value
    return known, needs_normalization


def _normalize_direct_solve_knowns(known_quantities: dict[str, ScalarValue]) -> set[str]:
    known = set()
    for key in get_core_batch_variable_keys():
        if key in known_quantities:
            known.add(key)

    if "feed_volume" in known_quantities and "feed_abv" in known_quantities:
        known.update({"W0", "x0"})
    elif "feed_abv" in known_quantities:
        known.add("x0")

    if "product_volume" in known_quantities and "product_abv" in known_quantities:
        known.update({"D", "xDavg"})
    elif "product_abv" in known_quantities:
        known.add("xDavg")

    if "bottoms_volume" in known_quantities and "bottoms_abv" in known_quantities:
        known.update({"B", "xB"})
    elif "bottoms_abv" in known_quantities:
        known.add("xB")

    return known


def _missing_descriptions(names: list[str]) -> list[str]:
    return [get_missing_value_display_name(name) for name in names]


def analyze_batch_capabilities(
    known_quantities: dict[str, ScalarValue],
    requested_outputs: list[str] | None = None,
    candidate_goals: list[str] | None = None,
) -> CapabilityAnalysis:
    """Analyze what direct-solve and batch-balance calculations are possible now."""

    requested_outputs = requested_outputs or []
    candidate_goals = candidate_goals or []
    normalized_knowns = _normalize_direct_solve_knowns(known_quantities)
    possible: list[CapabilityCalculationPath] = []

    rayleigh_paths = [
        CapabilityCalculationPath(
            calculation_id="rayleigh_known_w0_x0_xb",
            display_name="Rayleigh batch solve from W0, x0, and xB",
            known_inputs=["W0", "x0", "xB"],
            missing_inputs=[],
            outputs=["B", "D", "xDavg"],
            can_execute={"W0", "x0", "xB"}.issubset(normalized_knowns),
            explanation="Use the starting charge and stopping composition to solve one direct Rayleigh-constrained batch case.",
            suggested_next_question=None,
        ),
        CapabilityCalculationPath(
            calculation_id="rayleigh_known_w0_x0_d",
            display_name="Rayleigh batch solve from W0, x0, and D",
            known_inputs=["W0", "x0", "D"],
            missing_inputs=[],
            outputs=["B", "xB", "xDavg"],
            can_execute={"W0", "x0", "D"}.issubset(normalized_knowns),
            explanation="Use the starting charge and collected distillate amount to solve the stopping composition and remaining variables.",
            suggested_next_question=None,
        ),
        CapabilityCalculationPath(
            calculation_id="rayleigh_known_d_xdavg_x0",
            display_name="Rayleigh batch solve from D, xDavg, and x0",
            known_inputs=["D", "xDavg", "x0"],
            missing_inputs=[],
            outputs=["W0", "B", "xB"],
            can_execute={"D", "xDavg", "x0"}.issubset(normalized_knowns),
            explanation="Use the target distillate amount/composition and feed composition to solve the needed starting charge and stopping state.",
            suggested_next_question=None,
        ),
        CapabilityCalculationPath(
            calculation_id="rayleigh_known_d_xdavg_xb",
            display_name="Rayleigh batch solve from D, xDavg, and xB",
            known_inputs=["D", "xDavg", "xB"],
            missing_inputs=[],
            outputs=["W0", "B", "x0"],
            can_execute={"D", "xDavg", "xB"}.issubset(normalized_knowns),
            explanation="Use the target distillate amount/composition and stopping composition to solve the required starting feed state.",
            suggested_next_question=None,
        ),
        CapabilityCalculationPath(
            calculation_id="rayleigh_known_w0_x0_xdavg",
            display_name="Rayleigh batch solve from W0, x0, and xDavg",
            known_inputs=["W0", "x0", "xDavg"],
            missing_inputs=["xB or B/W or D or sweep permission"],
            outputs=["xB", "B/W", "D"],
            can_execute=False,
            explanation=(
                "From W0, x0, and xDavg, I know the starting charge and a target on the average distillate composition, "
                "but I still need one stopping basis to determine a unique direct Rayleigh case."
            ),
            suggested_next_question=(
                "Do you want to provide a final bottoms ABV/mole fraction, a final still amount B/W, a distillate amount D, or sweep over possible final bottoms compositions?"
            ),
        ),
    ]
    possible.extend(rayleigh_paths)

    mole_balance_paths = [
        CapabilityCalculationPath(
            calculation_id="mole_balance_known_w0_x0_d_xdavg",
            display_name="Mole-balance solve from W0, x0, D, and xDavg",
            known_inputs=["W0", "x0", "D", "xDavg"],
            missing_inputs=[],
            outputs=["B", "xB"],
            can_execute={"W0", "x0", "D", "xDavg"}.issubset(normalized_knowns),
            explanation="Solve the remaining still amount and composition from the total and ethanol balances.",
            suggested_next_question=None,
        ),
        CapabilityCalculationPath(
            calculation_id="mole_balance_known_w0_x0_b_xb",
            display_name="Mole-balance solve from W0, x0, B, and xB",
            known_inputs=["W0", "x0", "B", "xB"],
            missing_inputs=[],
            outputs=["D", "xDavg"],
            can_execute={"W0", "x0", "B", "xB"}.issubset(normalized_knowns),
            explanation="Solve the collected distillate amount and composition from the total and ethanol balances.",
            suggested_next_question=None,
        ),
        CapabilityCalculationPath(
            calculation_id="mole_balance_known_d_xdavg_b_xb",
            display_name="Mole-balance solve from D, xDavg, B, and xB",
            known_inputs=["D", "xDavg", "B", "xB"],
            missing_inputs=[],
            outputs=["W0", "x0"],
            can_execute={"D", "xDavg", "B", "xB"}.issubset(normalized_knowns),
            explanation="Solve the starting feed amount and composition from the total and ethanol balances.",
            suggested_next_question=None,
        ),
    ]
    possible.extend(mole_balance_paths)

    executable = [path for path in possible if path.can_execute]
    nearly_possible = []
    for path in possible:
        if path.can_execute:
            continue
        missing = [name for name in path.known_inputs if name not in normalized_knowns]
        if path.calculation_id == "rayleigh_known_w0_x0_xdavg" and {"W0", "x0", "xDavg"}.issubset(normalized_knowns):
            nearly_possible.append(path)
            continue
        if len(missing) <= 1:
            nearly_possible.append(
                path.model_copy(
                    update={
                        "missing_inputs": _missing_descriptions(missing) if missing else path.missing_inputs,
                        "suggested_next_question": (
                            f"If you provide {_missing_descriptions(missing)[0]}, I can run this direct solve."
                            if len(missing) == 1
                            else path.suggested_next_question
                        ),
                    }
                )
            )

    blocking_missing_values: list[str] = []
    for path in nearly_possible:
        for item in path.missing_inputs:
            if item not in blocking_missing_values:
                blocking_missing_values.append(item)

    suggested_next_questions = [
        path.suggested_next_question
        for path in nearly_possible
        if path.suggested_next_question
    ]
    preferred_next_question = suggested_next_questions[0] if suggested_next_questions else None

    if executable:
        reasoning_summary = "At least one supported direct-solve or batch-balance calculation is executable now."
    elif nearly_possible:
        reasoning_summary = "The request is in-domain and close to executable, but it needs one more value or operating choice."
    else:
        reasoning_summary = "The request is in-domain, but none of the currently supported direct-solve calculation patterns are fully matched yet."

    return CapabilityAnalysis(
        known_quantities=known_quantities,
        missing_quantities=blocking_missing_values,
        possible_calculations=possible,
        nearly_possible_calculations=nearly_possible,
        executable_calculations=executable,
        blocking_missing_values=blocking_missing_values,
        suggested_next_questions=suggested_next_questions,
        preferred_next_question=preferred_next_question,
        reasoning_summary=reasoning_summary,
        can_calculate_now=bool(executable),
    )


def analyze_intent_capabilities(
    user_message: str,
    classification: GoalClassification | None = None,
    plan: WorkflowPlan | None = None,
) -> IntentAnalysis:
    """Produce a soft capability analysis before strict workflow execution."""

    domain_relevant = _is_domain_relevant(user_message, classification)
    likely_task_family = _infer_likely_task_family(classification)
    known_quantities, needs_normalization = _collect_known_quantities(classification)
    requested_outputs = list(classification.requested_outputs) if classification is not None else []
    candidate_goals: list[str] = []
    if classification is not None and classification.goal != "unsupported_or_unclear":
        candidate_goals.append(classification.goal)

    capability_analysis = analyze_batch_capabilities(
        known_quantities=known_quantities,
        requested_outputs=requested_outputs,
        candidate_goals=candidate_goals,
    )

    possible_calculations = [path.explanation for path in capability_analysis.possible_calculations]
    missing_but_needed = list(capability_analysis.blocking_missing_values)
    best_next_question = capability_analysis.preferred_next_question
    explanation_hint: str | None = None
    can_calculate_now = bool(plan.ready_to_execute) if plan is not None else capability_analysis.can_calculate_now

    if domain_relevant and not can_calculate_now:
        if capability_analysis.nearly_possible_calculations:
            top = capability_analysis.nearly_possible_calculations[0]
            explanation_hint = f"{top.explanation} {top.suggested_next_question or ''}".strip()
        elif plan is not None and plan.missing_inputs:
            missing_but_needed = list(plan.missing_inputs)
            explanation_hint = (
                f"I recognize this as an in-domain batch distillation request, but I still need "
                f"{', '.join(missing_but_needed)} before I can run the current workflow."
            )
        else:
            explanation_hint = (
                "I recognize the request, but I still need one more operating constraint or stopping basis before I can run it."
            )
    elif not domain_relevant:
        explanation_hint = (
            "I’m not yet confident this is a batch-distillation or ethanol-water calculation request. "
            "If you want a distillation calculation, describe the known feed, product, or bottoms values and what you want to solve for."
        )

    return IntentAnalysis(
        raw_user_message=user_message,
        domain_relevant=domain_relevant,
        likely_task_family=likely_task_family,
        candidate_goals=candidate_goals,
        known_quantities=known_quantities,
        known_quantities_needing_normalization=needs_normalization,
        requested_outputs=requested_outputs,
        possible_calculations=possible_calculations,
        missing_but_needed=missing_but_needed,
        can_calculate_now=can_calculate_now,
        best_next_question=best_next_question,
        explanation_hint=explanation_hint,
        confidence=classification.confidence if classification is not None else None,
        capability_analysis=capability_analysis,
    )


def is_incomplete_workflow_followup(user_message: str) -> bool:
    normalized = _normalize_text(user_message)
    return any(term in normalized for term in _INCOMPLETE_FOLLOWUP_TERMS)


def build_incomplete_workflow_response(intent_analysis: IntentAnalysis) -> str:
    if intent_analysis.explanation_hint:
        return intent_analysis.explanation_hint

    if intent_analysis.capability_analysis and intent_analysis.capability_analysis.nearly_possible_calculations:
        top = intent_analysis.capability_analysis.nearly_possible_calculations[0]
        return f"{top.explanation} {top.suggested_next_question or ''}".strip()

    if intent_analysis.missing_but_needed:
        missing_text = ", ".join(intent_analysis.missing_but_needed)
        question = intent_analysis.best_next_question or "Tell me which missing value you want to provide next."
        return (
            f"I can see what you want to calculate, but I still need {missing_text}. "
            f"{question}"
        )

    return (
        "I recognize the request as batch-distillation related, but I still need one more operating detail "
        "before I can turn it into an executable workflow."
    )


def build_pending_incomplete_followup_response(
    user_message: str,
    intent_analysis: IntentAnalysis,
) -> str | None:
    if not is_incomplete_workflow_followup(user_message):
        return None

    normalized = _normalize_text(user_message)
    capability = intent_analysis.capability_analysis
    if capability is None:
        return build_incomplete_workflow_response(intent_analysis)

    if "what additional values do you need" in normalized or "what else do you need" in normalized or "what is missing" in normalized or "what values are missing" in normalized or "what more do you need" in normalized:
        if capability.blocking_missing_values:
            return (
                "The additional value or choice that would make this executable is: "
                f"{', '.join(capability.blocking_missing_values)}."
            )

    if "what are my options" in normalized or "what can you calculate from that" in normalized or "what can you calculate from these" in normalized:
        option_lines = []
        for path in capability.nearly_possible_calculations[:3]:
            missing_text = ", ".join(path.missing_inputs) if path.missing_inputs else "nothing else"
            option_lines.append(f"- {path.display_name}: missing {missing_text}")
        for path in capability.executable_calculations[:2]:
            option_lines.append(f"- {path.display_name}: executable now")
        if option_lines:
            return "Here are the most relevant options right now:\n" + "\n".join(option_lines)

    if "can you sweep it instead" in normalized:
        if intent_analysis.likely_task_family == "direct_rayleigh" and {"W0", "x0", "xDavg"}.issubset(set(intent_analysis.known_quantities)):
            return (
                "For this case, a useful next step would be to sweep over possible final bottoms compositions xB and see the corresponding product and remaining-still outcomes. "
                "If you want that behavior, tell me to sweep over possible final bottoms compositions."
            )
        return (
            "If you want a sweep instead of one direct solve, tell me which variable you want to vary and I’ll guide you to the closest supported sweep workflow."
        )

    if "can you solve it now" in normalized:
        if capability.can_calculate_now:
            return "Yes, at least one supported direct calculation path is executable now."
        if capability.preferred_next_question:
            return f"Not yet. {capability.preferred_next_question}"

    if capability.preferred_next_question:
        return f"{capability.reasoning_summary} {capability.preferred_next_question}"

    return (
        "I recognize the workflow, but I still need one more operating constraint or stopping basis before I can run it."
    )
