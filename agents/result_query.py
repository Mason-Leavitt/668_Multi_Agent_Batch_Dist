"""Deterministic follow-up queries against the most recent execution result."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field
from .execution_schemas import WorkflowExecutionResult


RESULT_FOLLOWUP_TERMS = [
    "based on these results",
    "based on the results",
    "from the table",
    "from the plot",
    "in the results",
    "according to this run",
    "for this sweep",
    "how much can i get if",
    "what row",
    "what value",
    "if i want the composition to be",
    "if i want xdavg",
    "what d",
    "how much distillate",
]


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


class ResultQueryAnswer(BaseModel):
    """Structured deterministic answer for result-table follow-up questions."""

    answer: str
    matched_row: dict[str, Any] | None = None
    matched_column: str | None = None
    target_value: float | None = None
    returned_columns: list[str] = Field(default_factory=list)


def _format_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def is_result_followup_question(user_message: str) -> bool:
    normalized = _normalize_text(user_message)
    return any(term in normalized for term in RESULT_FOLLOWUP_TERMS)


def refers_to_matched_row(user_message: str) -> bool:
    normalized = _normalize_text(user_message)
    terms = [
        "that row",
        "that result",
        "same row",
        "for that one",
        "for it",
        "for that row",
    ]
    return any(term in normalized for term in terms)


def extract_abv_target_percent(user_message: str) -> float | None:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*%\s*abv",
        r"(\d+(?:\.\d+)?)\s*percent\s*abv",
        r"(\d+(?:\.\d+)?)\s*percent",
        r"(\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def find_closest_row_by_column(
    rows: list[dict],
    column: str,
    target: float,
) -> dict | None:
    best_row: dict | None = None
    best_distance: float | None = None

    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        distance = abs(numeric_value - target)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_row = row

    return best_row


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


FEED_SIDE_MATCH_TERMS = [
    "input composition",
    "input concentration",
    "input abv",
    "input strength",
    "input composition is",
    "input composition of",
    "feed composition",
    "feed concentration",
    "feed abv",
    "feed strength",
    "feed composition is",
    "feed composition of",
    "starting composition",
    "starting concentration",
    "starting abv",
    "wash composition",
    "wash concentration",
    "wash abv",
    "x0",
]

FEED_SIDE_RETURN_TERMS = [
    "how much feed",
    "feed volume",
    "feed amount",
    "amount of feed",
    "input volume",
    "input amount",
    "amount of input",
    "starting volume",
    "starting amount",
    "wash volume",
    "wash amount",
    "w0 volume",
    "w0_volume_l",
    "w0",
]

FEED_ABV_COLUMN_CANDIDATES = [
    "x0_abv_percent",
    "feed_abv_percent",
    "feed_abv",
    "input_abv_percent",
    "input_abv",
]

FEED_VOLUME_COLUMN_CANDIDATES = [
    "W0_volume_L",
    "feed_volume_L",
    "input_volume_L",
    "W0",
]


def _first_existing_column(rows: list[dict], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if any(row.get(candidate) is not None for row in rows):
            return candidate
    return None


def _get_value(row: dict, candidates: list[str]) -> tuple[str | None, object | None]:
    for candidate in candidates:
        value = row.get(candidate)
        if value is not None:
            return candidate, value
    return None, None


def _build_feed_to_product_answer(target_abv: float, matched_row: dict) -> ResultQueryAnswer | None:
    distillate_volume = matched_row.get("D_volume_L")
    if distillate_volume is None:
        return None

    answer = (
        f"Based on the current feed-to-product sweep, the closest row to {_format_float(target_abv, 2)}% ABV "
        f"has xDavg_abv_percent = {_format_float(matched_row.get('xDavg_abv_percent'), 2)}% "
        f"and D_volume_L = {_format_float(distillate_volume, 2)} L."
    )

    details: list[str] = []
    if matched_row.get("xDavg") is not None:
        details.append(f"xDavg = {_format_float(matched_row.get('xDavg'), 5)}")
    if matched_row.get("D") is not None:
        details.append(f"D = {_format_float(matched_row.get('D'), 2)} mol")
    if matched_row.get("xB_abv_percent") is not None:
        details.append(f"xB_abv_percent = {_format_float(matched_row.get('xB_abv_percent'), 2)}%")

    if details:
        answer += " The corresponding internal values are " + ", ".join(details) + "."

    answer += " This is the nearest row from the current results, not an interpolated value."
    return ResultQueryAnswer(
        answer=answer,
        matched_row=matched_row,
        matched_column="xDavg_abv_percent",
        target_value=target_abv,
        returned_columns=["D_volume_L", "D", "xDavg", "xB_abv_percent"],
    )


def _build_product_to_feed_feed_side_answer(target_abv: float, matched_row: dict) -> ResultQueryAnswer | None:
    matched_column, matched_abv_value = _get_value(matched_row, FEED_ABV_COLUMN_CANDIDATES)
    return_column, feed_volume = _get_value(matched_row, FEED_VOLUME_COLUMN_CANDIDATES)
    returned_columns: list[str] = []
    if matched_abv_value is None:
        return None
    if feed_volume is None:
        return None
    if return_column is not None:
        returned_columns.append(return_column)
    if matched_row.get("W0") is not None and "W0" not in returned_columns:
        returned_columns.append("W0")

    answer = (
        f"Based on the current product-to-feed sweep, the closest row to a "
        f"{_format_float(target_abv, 2)}% feed/input ABV has {matched_column} = "
        f"{_format_float(matched_abv_value, 2)}%."
    )
    if return_column == "W0_volume_L":
        answer += f" That row requires W0_volume_L = {_format_float(feed_volume, 2)} L of feed."
    elif return_column in {"feed_volume_L", "input_volume_L"}:
        answer += f" That row requires {return_column} = {_format_float(feed_volume, 2)} L of feed."
    elif matched_row.get("W0") is not None:
        answer += f" That row requires W0 = {_format_float(matched_row.get('W0'), 2)} mol of feed."

    details: list[str] = []
    if matched_row.get("W0") is not None:
        details.append(f"W0 = {_format_float(matched_row.get('W0'), 2)} mol")
    if matched_row.get("x0") is not None:
        details.append(f"x0 = {_format_float(matched_row.get('x0'), 5)}")
    if matched_row.get("xB_abv_percent") is not None:
        details.append(f"xB_abv_percent = {_format_float(matched_row.get('xB_abv_percent'), 2)}%")

    if details:
        answer += " Internally, " + ", ".join(details) + "."

    answer += " This is the nearest row in the sweep, not an interpolated value."
    return ResultQueryAnswer(
        answer=answer,
        matched_row=matched_row,
        matched_column=matched_column,
        target_value=target_abv,
        returned_columns=returned_columns,
    )


def _answer_product_to_feed_question(normalized: str, target_abv: float, execution_result: WorkflowExecutionResult) -> ResultQueryAnswer | None:
    wants_feed_side_match = _contains_any(normalized, FEED_SIDE_MATCH_TERMS)
    wants_feed_side_return = _contains_any(normalized, FEED_SIDE_RETURN_TERMS)

    if wants_feed_side_match and wants_feed_side_return:
        match_column = _first_existing_column(execution_result.rows, FEED_ABV_COLUMN_CANDIDATES)
        if match_column is None:
            return ResultQueryAnswer(
                answer=(
                    "I found the current product-to-feed result table, but it does not include a feed ABV "
                    "column such as x0_abv_percent, so I cannot perform that lookup yet."
                ),
                matched_row=None,
                matched_column=None,
                target_value=target_abv,
                returned_columns=[],
            )

        matched_row = find_closest_row_by_column(execution_result.rows, match_column, target_abv)
        if matched_row is None:
            return None

        if _first_existing_column([matched_row], FEED_VOLUME_COLUMN_CANDIDATES) is None:
            return ResultQueryAnswer(
                answer=(
                    "I found the closest feed composition row, but the result table does not include "
                    "W0_volume_L or another feed-volume column to report."
                ),
                matched_row=matched_row,
                matched_column=match_column,
                target_value=target_abv,
                returned_columns=[],
            )

        return _build_product_to_feed_feed_side_answer(target_abv, matched_row)

    return None


def _answer_followup_about_matched_row(user_message: str, matched_row: dict | None) -> ResultQueryAnswer | None:
    if not matched_row:
        return None

    normalized = _normalize_text(user_message)

    if _contains_any(normalized, FEED_SIDE_RETURN_TERMS):
        return_column, feed_volume = _get_value(matched_row, FEED_VOLUME_COLUMN_CANDIDATES)
        if feed_volume is not None and return_column == "W0_volume_L":
            answer = f"For that matched row, the feed volume is W0_volume_L = {_format_float(feed_volume, 2)} L."
            if matched_row.get("W0") is not None:
                answer += f" The internal W0 value is {_format_float(matched_row.get('W0'), 2)} mol."
            return ResultQueryAnswer(
                answer=answer,
                matched_row=matched_row,
                returned_columns=["W0_volume_L", "W0"],
            )
        if feed_volume is not None and return_column in {"feed_volume_L", "input_volume_L"}:
            answer = f"For that matched row, the feed volume is {return_column} = {_format_float(feed_volume, 2)} L."
            if matched_row.get("W0") is not None:
                answer += f" The internal W0 value is {_format_float(matched_row.get('W0'), 2)} mol."
            return ResultQueryAnswer(
                answer=answer,
                matched_row=matched_row,
                returned_columns=[return_column, "W0"],
            )
        if matched_row.get("W0") is not None:
            return ResultQueryAnswer(
                answer=f"For that matched row, the feed amount is W0 = {_format_float(matched_row.get('W0'), 2)} mol.",
                matched_row=matched_row,
                returned_columns=["W0"],
            )

    if _contains_any(normalized, FEED_SIDE_MATCH_TERMS):
        abv_column, abv_value = _get_value(matched_row, FEED_ABV_COLUMN_CANDIDATES)
        if abv_value is not None and abv_column is not None:
            answer = f"For that matched row, the feed ABV is {abv_column} = {_format_float(abv_value, 2)}%."
            if matched_row.get("x0") is not None:
                answer += f" The internal x0 value is {_format_float(matched_row.get('x0'), 5)}."
            return ResultQueryAnswer(
                answer=answer,
                matched_row=matched_row,
                returned_columns=[abv_column, "x0"],
            )

    if "xb" in normalized or "bottoms" in normalized:
        if "abv" in normalized and matched_row.get("xB_abv_percent") is not None:
            return ResultQueryAnswer(
                answer=f"For that matched row, xB_abv_percent = {_format_float(matched_row.get('xB_abv_percent'), 2)}%.",
                matched_row=matched_row,
                returned_columns=["xB_abv_percent"],
            )
        if matched_row.get("xB") is not None:
            return ResultQueryAnswer(
                answer=f"For that matched row, xB = {_format_float(matched_row.get('xB'), 5)}.",
                matched_row=matched_row,
                returned_columns=["xB"],
            )

    if "error" in normalized and matched_row.get("error") is not None:
        return ResultQueryAnswer(
            answer=f"For that matched row, error = {_format_float(matched_row.get('error'), 5)}.",
            matched_row=matched_row,
            returned_columns=["error"],
        )

    return None


def answer_result_question(
    user_message: str,
    execution_result: WorkflowExecutionResult,
) -> ResultQueryAnswer | None:
    if not execution_result.rows:
        return None

    normalized = _normalize_text(user_message)
    target_abv = extract_abv_target_percent(user_message)
    if target_abv is None:
        return None

    if execution_result.workflow_name == "product_to_feed_sweep":
        answer = _answer_product_to_feed_question(normalized, target_abv, execution_result)
        if answer is not None:
            return answer

    if execution_result.workflow_name == "feed_to_product_sweep":
        matched_row = find_closest_row_by_column(
            execution_result.rows,
            "xDavg_abv_percent",
            target_abv,
        )
        if matched_row is None:
            return None
        return _build_feed_to_product_answer(target_abv, matched_row)

    return None


def answer_followup_about_matched_row(
    user_message: str,
    matched_row: dict | None,
) -> ResultQueryAnswer | None:
    return _answer_followup_about_matched_row(user_message, matched_row)
