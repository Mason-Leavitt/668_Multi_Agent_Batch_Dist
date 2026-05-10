"""Primary and fallback classification paths for the interface agent prototype."""

from __future__ import annotations

import re

from .schemas import GoalClassification, OutputFormat, OutputMode


VARIABLE_ALIASES = {
    "W0": ["w0", "initial moles", "starting moles", "starting charge", "feed amount"],
    "x0": ["x0", "initial composition", "initial mole fraction", "feed composition"],
    "D": [" d ", "distillate amount", "product amount", "amount of product"],
    "xDavg": [
        "xdavg",
        "average distillate composition",
        "average product composition",
        "product strength",
        "distillate strength",
    ],
    "B": [" b ", "bottoms amount", "remaining amount", "amount left"],
    "xB": [
        "xb",
        "bottoms composition",
        "boiler composition",
        "stopping composition",
        "final pot composition",
    ],
    "feed_volume": ["feed volume", "wash volume", "starting volume", "charge volume"],
    "feed_abv": ["feed abv", "wash abv", "starting abv", "feed strength", "wash strength"],
    "product_volume": [
        "product volume",
        "distillate volume",
        "product amount",
        "distillate amount",
    ],
    "product_abv": [
        "product abv",
        "distillate abv",
        "product strength",
        "distillate strength",
    ],
}

DISPLAY_VARIABLE_ORDER = [
    "W0",
    "x0",
    "D",
    "xDavg",
    "B",
    "xB",
    "feed_volume",
    "feed_abv",
    "product_volume",
    "product_abv",
]

VOLUME_TERMS = [
    "gallon",
    "gallons",
    "gal",
    "liter",
    "liters",
    "litre",
    "litres",
    " ml ",
    "milliliter",
    "milliliters",
]
ABV_TERMS = [
    "abv",
    "% abv",
    "percent alcohol",
    "alcohol by volume",
    "proof",
]
EXPLANATION_TERMS = ["explain", "what is", "what does", "why", "how does"]
PLOT_TERMS = ["plot", "graph", "chart", "visualize"]
TABLE_TERMS = ["table", "tabulate", "spreadsheet"]
CONSISTENCY_TERMS = [
    "make sense",
    "consistent",
    "physically valid",
    "valid together",
    "sanity check",
    "check whether",
]
MOLE_BALANCE_TERMS = ["mole balance", "material balance", "overall balance"]
RAYLEIGH_TERMS = ["rayleigh", "rayleigh equation"]
SWEEP_HINTS = ["sweep", "range", "over", "different", "vary", "plot"]
PRODUCT_TARGET_TERMS = [
    "want",
    "need",
    "target",
    "required",
    "required feed",
    "what feed do i need",
]


def _normalize_text(text: str) -> str:
    normalized = text.casefold()
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"([,?.!;:()])", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f" {normalized} "


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _has_standalone_variable(text: str, variable: str) -> bool:
    patterns = {
        "W0": r"\bw0\b",
        "x0": r"\bx0\b",
        "D": r"(?<!x)\bd\b(?!avg)",
        "xDavg": r"\bxdavg\b",
        "B": r"(?<!x)\bb\b(?!\w)",
        "xB": r"\bxb\b",
    }
    return bool(re.search(patterns[variable], text))


def _detect_known_inputs(text: str) -> list[str]:
    known: list[str] = []

    if _has_standalone_variable(text, "W0"):
        known.append("W0")
    if _has_standalone_variable(text, "x0"):
        known.append("x0")
    if _has_standalone_variable(text, "D"):
        known.append("D")
    if _has_standalone_variable(text, "xDavg"):
        known.append("xDavg")
    if _has_standalone_variable(text, "B"):
        known.append("B")
    if _has_standalone_variable(text, "xB"):
        known.append("xB")

    if _contains_any(text, ["starting with", "i have", "given a feed volume", "wash"]):
        if _contains_volume_value(text):
            known.append("feed_volume")
        if _contains_abv_value(text):
            known.append("feed_abv")

    if _contains_any(text, ["i want", "target", "need to get", "product at", "distillate at"]):
        if _contains_volume_value(text):
            known.append("product_volume")
        if _contains_abv_value(text):
            known.append("product_abv")

    for canonical, aliases in VARIABLE_ALIASES.items():
        if canonical in {"D", "B", "W0", "x0", "xDavg", "xB"}:
            continue
        if canonical in known:
            continue
        if _contains_any(text, aliases):
            known.append(canonical)

    return _dedupe(known)


def _detect_requested_outputs(text: str) -> list[str]:
    requested: list[str] = []

    if _contains_any(text, ["what d", " d and ", "product amount", "distillate amount", "product outcomes"]):
        requested.append("D")
    if _contains_any(
        text,
        ["xdavg", "product strength", "distillate strength", "average distillate composition"],
    ):
        requested.append("xDavg")
    if _contains_any(text, ["what product amount and strength", "distillate outcomes", "possible distillate outcomes"]):
        requested.extend(["D", "xDavg"])
    if _contains_any(text, ["what w0", "possible w0", "starting feed", "feed is required"]):
        requested.append("W0")
    if _contains_any(text, ["what x0", "possible x0", "feed composition", "what abv do i need"]):
        requested.append("x0")
    if _contains_any(text, ["feed do i need", "how much wash", "starting volume"]):
        requested.append("feed_volume")
    if _contains_any(text, ["what abv do i need", "wash at what abv", "feed abv"]):
        requested.append("feed_abv")
    if _contains_any(text, ["solve for b", "find b", "what is b"]):
        requested.append("B")
    if _contains_any(text, ["solve for xb", "find xb", "what is xb"]):
        requested.append("xB")
    if _contains_any(text, ["solve for x0", "find x0", "what is x0"]):
        requested.append("x0")
    if _contains_any(text, ["solve for w0", "find w0", "what is w0"]):
        requested.append("W0")

    if _contains_any(text, ["explain the rayleigh equation", "what is xdavg", "what does w0 mean"]):
        if "rayleigh" in text:
            requested.append("Rayleigh equation")
        if "xdavg" in text:
            requested.append("xDavg")
        if "w0" in text:
            requested.append("W0")

    return _dedupe(requested)


def _detect_output_mode(text: str) -> OutputMode:
    wants_plot = _contains_any(text, PLOT_TERMS)
    wants_table = _contains_any(text, TABLE_TERMS)
    wants_explanation = _contains_any(text, EXPLANATION_TERMS)

    signals = sum([wants_plot, wants_table, wants_explanation])
    if signals > 1:
        return "mixed"
    if wants_plot:
        return "plot"
    if wants_table:
        return "table"
    if wants_explanation:
        return "explanation"
    return "numeric_answer"


def _contains_volume_value(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d+(\.\d+)?\s*(gallon|gallons|gal|liter|liters|litre|litres|l|ml)\b",
            text,
        )
    )


def _contains_abv_value(text: str) -> bool:
    return bool(
        re.search(r"\b\d+(\.\d+)?\s*(%|proof)\s*(abv)?\b", text)
        or re.search(r"\b\d+(\.\d+)?\s*abv\b", text)
    )


def _extract_named_value(raw_text: str, variable: str) -> str | None:
    patterns = {
        "W0": r"\bW0\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        "x0": r"\bx0\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        "D": r"(?<!x)\bD\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        "xDavg": r"\bxDavg\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        "B": r"(?<!x)\bB\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        "xB": r"\bxB\b\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
    }
    match = re.search(patterns[variable], raw_text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _extract_volume_assignment(raw_text: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(gallons?|gal|liters?|litres?|litres|liter|litre|l|ml)\b",
        raw_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _extract_abv_assignment(raw_text: str) -> str | None:
    match = re.search(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(%?\s*ABV|ABV|proof)\b",
        raw_text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*%\b", raw_text, flags=re.IGNORECASE)
        if not match:
            return None
        return f"{match.group(1)}%"
    value = re.sub(r"\s+", " ", match.group(0).strip())
    return value


def _extract_variable_assignments(raw_text: str, normalized_text: str, goal: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    refers_to_feed = _contains_any(normalized_text, ["wash", "feed", "starting with", "i have"])
    refers_to_product = _contains_any(normalized_text, ["product", "distillate", "to get", "i want"])

    for variable in ["W0", "x0", "D", "xDavg", "B", "xB"]:
        value = _extract_named_value(raw_text, variable)
        if value is not None:
            assignments[variable] = value

    volume_value, volume_unit = _extract_volume_assignment(raw_text)
    abv_value = _extract_abv_assignment(raw_text)

    if volume_value and volume_unit:
        volume_display = f"{volume_value} {volume_unit}"
        if goal == "feed_to_product_sweep" and refers_to_feed:
            assignments["feed_volume"] = volume_display
        elif goal == "product_to_feed_sweep" or (refers_to_product and not refers_to_feed):
            assignments["product_volume"] = volume_display
        elif refers_to_feed:
            assignments["feed_volume"] = volume_display
        else:
            assignments["feed_volume"] = volume_display

    if abv_value:
        if goal == "feed_to_product_sweep" and refers_to_feed:
            assignments["feed_abv"] = abv_value
        elif goal == "product_to_feed_sweep" or (refers_to_product and not refers_to_feed):
            assignments["product_abv"] = abv_value
        elif refers_to_feed:
            assignments["feed_abv"] = abv_value
        else:
            assignments["feed_abv"] = abv_value

    return {
        key: assignments[key]
        for key in DISPLAY_VARIABLE_ORDER
        if key in assignments
    }


def _detect_input_format(text: str) -> str:
    has_model = any(_has_standalone_variable(text, variable) for variable in ["W0", "x0", "D", "xDavg", "B", "xB"])
    has_user_units = _contains_any(text, VOLUME_TERMS) or _contains_any(text, ABV_TERMS)

    if has_model and has_user_units:
        return "mixed_units"
    if has_user_units:
        return "volume_abv"
    if has_model:
        return "model_units"
    return "unknown"


def _detect_output_format(text: str, input_format: str) -> OutputFormat:
    has_user_units = _contains_any(text, VOLUME_TERMS) or _contains_any(text, ABV_TERMS)
    has_model = any(_has_standalone_variable(text, variable) for variable in ["W0", "x0", "D", "xDavg", "B", "xB"])

    if has_model and has_user_units:
        return "mixed_units"
    if has_user_units:
        return "volume_abv"
    if _contains_any(text, ["amount", "strength", "outcomes", "required feed"]):
        return "user_friendly"
    if has_model:
        return "model_units"
    if input_format == "volume_abv":
        return "user_friendly"
    return "unknown"


def _detect_sweep_variable(text: str) -> str | None:
    if not _contains_any(text, SWEEP_HINTS):
        return None
    if "xb" in text or "stopping composition" in text or "final pot composition" in text:
        return "xB"
    if "x0" in text or "feed composition" in text:
        return "x0"
    return None


def _determine_missing_inputs(goal: str, known_inputs: list[str], sweep_variable: str | None) -> list[str]:
    required_by_goal = {
        "feed_to_product_sweep": ["W0", "x0"],
        "product_to_feed_sweep": ["D", "xDavg"],
        "single_rayleigh_calculation": ["W0", "x0", "xB"],
        "mole_balance_calculation": [],
        "consistency_check": [],
        "explain_variable_or_workflow": [],
        "unsupported_or_unclear": [],
    }

    required = list(required_by_goal.get(goal, []))
    if goal == "feed_to_product_sweep" and {"feed_volume", "feed_abv"}.issubset(set(known_inputs)):
        required = ["feed_volume", "feed_abv"]
    if goal == "product_to_feed_sweep" and {"product_volume", "product_abv"}.issubset(set(known_inputs)):
        required = ["product_volume", "product_abv"]
    if sweep_variable and sweep_variable not in required and goal in {
        "feed_to_product_sweep",
        "product_to_feed_sweep",
    }:
        required.append(sweep_variable)

    return [name for name in required if name not in known_inputs]


def _postprocess_for_goal(
    goal: str,
    text: str,
    known_inputs: list[str],
    requested_outputs: list[str],
    input_format: str,
    output_format: OutputFormat,
) -> tuple[list[str], list[str], OutputFormat]:
    known = list(known_inputs)
    requested = list(requested_outputs)
    final_output_format = output_format

    if goal == "feed_to_product_sweep":
        requested = _dedupe(requested + [name for name in ["D", "xDavg"] if name in requested or input_format == "volume_abv"])
        if not _contains_any(text, ["given d", "know d", "with d", "given xdavg", "know xdavg", "with xdavg"]):
            known = [name for name in known if name not in {"D", "xDavg"}]
        if input_format == "volume_abv":
            known = [name for name in known if name not in {"product_volume", "product_abv"}]
            final_output_format = "user_friendly"

    if goal == "product_to_feed_sweep":
        if not _contains_any(text, ["given w0", "know w0", "with w0", "given x0", "know x0", "with x0"]):
            known = [name for name in known if name not in {"W0", "x0"}]
        if input_format == "volume_abv":
            requested = _dedupe(requested + ["feed_volume", "feed_abv"])
            final_output_format = "user_friendly"
        elif any(name in known for name in ["D", "xDavg"]):
            requested = _dedupe(requested + ["W0", "x0"])

    return known, requested, final_output_format


def _build_summary(
    goal: str,
    output_mode: OutputMode,
    input_format: str,
    output_format: OutputFormat,
    known_inputs: list[str],
    requested_outputs: list[str],
) -> tuple[str, str]:
    reasoning = (
        f"Detected goal '{goal}' from deterministic keyword rules. "
        f"Known inputs: {known_inputs or ['none detected']}. "
        f"Requested outputs: {requested_outputs or ['none detected']}. "
        f"Input format is '{input_format}' and output format is '{output_format}'."
    )
    user_summary = (
        f"This looks like a '{goal}' request. "
        f"The prototype currently classifies the request only and would present the result as '{output_mode}'."
    )
    return reasoning, user_summary


def classify_goal_deterministic(user_message: str) -> GoalClassification:
    """Classify a user request into the prototype goal schema using rules only."""

    normalized = _normalize_text(user_message)
    known_inputs = _detect_known_inputs(normalized)
    requested_outputs = _detect_requested_outputs(normalized)
    input_format = _detect_input_format(normalized)
    output_format = _detect_output_format(normalized, input_format)
    output_mode = _detect_output_mode(normalized)
    sweep_variable = _detect_sweep_variable(normalized)

    goal = "unsupported_or_unclear"
    confidence = 0.2

    if _contains_any(normalized, EXPLANATION_TERMS):
        goal = "explain_variable_or_workflow"
        output_mode = "explanation"
        confidence = 0.92
    elif _contains_any(normalized, CONSISTENCY_TERMS):
        goal = "consistency_check"
        output_mode = "mixed" if output_mode in {"plot", "table"} else "explanation"
        confidence = 0.88
    elif _contains_any(normalized, MOLE_BALANCE_TERMS) or (
        "solve for" in normalized and _contains_any(normalized, [" x0 ", " xb ", " b ", " w0 "])
    ):
        goal = "mole_balance_calculation"
        output_mode = "numeric_answer" if output_mode == "numeric_answer" else output_mode
        confidence = 0.9
    elif "rayleigh" in normalized and not _contains_any(normalized, ["sweep", "possible", "combinations"]):
        goal = "single_rayleigh_calculation"
        output_mode = "numeric_answer" if output_mode == "numeric_answer" else output_mode
        confidence = 0.84
    else:
        wants_product_outcomes = any(name in known_inputs for name in ["W0", "x0", "feed_volume", "feed_abv"]) and (
            any(name in requested_outputs for name in ["D", "xDavg"])
            or _contains_any(normalized, ["what product", "distillate outcomes", "could i get", "product outcomes"])
        )
        wants_feed_requirements = any(
            name in known_inputs for name in ["D", "xDavg", "product_volume", "product_abv"]
        ) and (
            any(name in requested_outputs for name in ["W0", "x0", "feed_volume", "feed_abv"])
            or _contains_any(normalized, ["what feed do i need", "feed is required", "starting feed"])
        )

        if wants_product_outcomes:
            goal = "feed_to_product_sweep"
            confidence = 0.91 if input_format != "unknown" else 0.78
        elif wants_feed_requirements or (
            _contains_any(normalized, PRODUCT_TARGET_TERMS)
            and any(name in known_inputs for name in ["product_volume", "product_abv", "D", "xDavg"])
        ):
            goal = "product_to_feed_sweep"
            confidence = 0.9 if input_format != "unknown" else 0.76

    requires_input_conversion = input_format in {"volume_abv", "mixed_units"}
    requires_output_conversion = False

    if goal in {"feed_to_product_sweep", "product_to_feed_sweep"}:
        if output_format in {"volume_abv", "mixed_units", "user_friendly"} and input_format != "model_units":
            requires_output_conversion = True
        elif output_format in {"volume_abv", "mixed_units"}:
            requires_output_conversion = True

    missing_inputs = _determine_missing_inputs(goal, known_inputs, sweep_variable)
    known_inputs, requested_outputs, output_format = _postprocess_for_goal(
        goal=goal,
        text=normalized,
        known_inputs=known_inputs,
        requested_outputs=requested_outputs,
        input_format=input_format,
        output_format=output_format,
    )
    missing_inputs = _determine_missing_inputs(goal, known_inputs, sweep_variable)
    reasoning_summary, user_facing_summary = _build_summary(
        goal=goal,
        output_mode=output_mode,
        input_format=input_format,
        output_format=output_format,
        known_inputs=known_inputs,
        requested_outputs=requested_outputs,
    )
    variable_assignments = _extract_variable_assignments(user_message, normalized, goal)

    if goal == "unsupported_or_unclear":
        user_facing_summary = (
            "The prototype needs a clearer calculation goal before it can route this request. "
            "Try stating the known inputs, the desired outputs, and whether you want an explanation or a calculation."
        )

    return GoalClassification(
        goal=goal,
        output_mode=output_mode,
        known_inputs=known_inputs,
        requested_outputs=requested_outputs,
        missing_inputs=missing_inputs,
        variable_assignments=variable_assignments,
        input_format=input_format,
        output_format=output_format,
        requires_input_conversion=requires_input_conversion,
        requires_output_conversion=requires_output_conversion,
        sweep_variable=sweep_variable,
        confidence=confidence,
        reasoning_summary=reasoning_summary,
        user_facing_summary=user_facing_summary,
    )


def classify_goal_llm(user_message: str, model_name: str = "gpt-4o-mini") -> GoalClassification:
    """Classify a user request with an LLM using structured output."""

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    from .classification_prompt import CLASSIFICATION_SYSTEM_PROMPT

    load_dotenv()
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(GoalClassification)
    result = structured_llm.invoke(
        [
            ("system", CLASSIFICATION_SYSTEM_PROMPT),
            ("human", user_message),
        ]
    )
    return result


def classify_goal(
    user_message: str,
    use_llm: bool = True,
    model_name: str = "gpt-4o-mini",
) -> GoalClassification:
    """Classify a user request with the primary LLM path and deterministic fallback."""

    if use_llm:
        try:
            return classify_goal_llm(user_message, model_name=model_name)
        except Exception as exc:
            fallback = classify_goal_deterministic(user_message)
            fallback.reasoning_summary = (
                "Fallback used after LLM error: "
                f"{exc}. Deterministic baseline classification returned. "
                f"{fallback.reasoning_summary}"
            )
            fallback.user_facing_summary = (
                "The LLM classifier was unavailable, so the deterministic fallback was used."
            )
            return fallback

    return classify_goal_deterministic(user_message)
