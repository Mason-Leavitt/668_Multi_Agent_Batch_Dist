"""Chat-based Streamlit UI for the batch distillation interface agent prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engineering.calculations import (
    create_plot_D_vs_xDavg_in_L_ABV,
    create_plot_W0_vs_x0_combinations_in_L_ABV,
)

from agents.conversation_responder import respond_to_general_message
from agents.conversation_router import (
    extract_volume_abv_pair,
    infer_likely_goal,
    infer_target_role,
    route_conversation_message,
    safe_confirmation_fallback,
)
from agents.interface_agent import classify_goal
from agents.result_explainer import explain_execution_result
from agents.result_query import (
    answer_result_question,
    answer_followup_about_matched_row,
    extract_abv_target_percent,
    find_closest_row_by_column,
    refers_to_matched_row,
)
from agents.workflow_executor import execute_workflow
from agents.workflow_planner import create_workflow_plan

CHAT_CONTAINER = None


def stream_text(text: str):
    words = text.split(" ")
    for word in words:
        yield word + " "


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def add_user_message(content: str) -> None:
    _append_message("user", content)
    if CHAT_CONTAINER is not None:
        with CHAT_CONTAINER:
            with st.chat_message("user"):
                st.write(content)


def add_assistant_message(content: str, stream: bool = True) -> None:
    rendered_content = content
    if CHAT_CONTAINER is not None:
        with CHAT_CONTAINER:
            with st.chat_message("assistant"):
                if stream:
                    rendered_content = st.write_stream(stream_text(content))
                else:
                    st.markdown(content)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": rendered_content,
        }
    )


def _clear_pending_intent() -> None:
    st.session_state.pending_intent_hint = None
    st.session_state.pending_intent_context = None


def _update_recent_request_context(user_message: str) -> None:
    previous = st.session_state.recent_request_context or {}
    pair = extract_volume_abv_pair(user_message)
    likely_goal = infer_likely_goal(user_message)
    target_role = infer_target_role(user_message)

    st.session_state.recent_request_context = {
        "last_user_message": user_message,
        "detected_product_volume": (
            pair["volume"] if pair and target_role == "product" else previous.get("detected_product_volume")
        ),
        "detected_product_abv": (
            pair["abv"] if pair and target_role == "product" else previous.get("detected_product_abv")
        ),
        "detected_feed_volume": (
            pair["volume"] if pair and target_role == "feed" else previous.get("detected_feed_volume")
        ),
        "detected_feed_abv": (
            pair["abv"] if pair and target_role == "feed" else previous.get("detected_feed_abv")
        ),
        "likely_goal": likely_goal or previous.get("likely_goal"),
    }


def _build_contextual_classification_message(
    user_message: str,
    likely_goal: str | None = None,
    use_previous_context: bool = False,
) -> str:
    likely_goal = likely_goal or infer_likely_goal(user_message)
    recent_context = st.session_state.recent_request_context

    if use_previous_context and recent_context:
        if recent_context.get("detected_product_volume") and recent_context.get("detected_product_abv"):
            return (
                f"The user wants a {likely_goal or 'product_to_feed_sweep'}. "
                f"Recent product target context: {recent_context['detected_product_volume']} at "
                f"{recent_context['detected_product_abv']}. "
                f"Current message: {user_message}"
            )
        if recent_context.get("detected_feed_volume") and recent_context.get("detected_feed_abv"):
            return (
                f"The user wants a {likely_goal or 'feed_to_product_sweep'}. "
                f"Recent feed context: {recent_context['detected_feed_volume']} at "
                f"{recent_context['detected_feed_abv']}. "
                f"Current message: {user_message}"
            )

    if likely_goal == "product_to_feed_sweep":
        return (
            "Routing hint: this looks like product_to_feed_sweep because the user gave a desired "
            "product/output target and asked what feed or inputs could achieve it.\n"
            f"User request: {user_message}"
        )

    if likely_goal == "feed_to_product_sweep":
        return (
            "Routing hint: this looks like feed_to_product_sweep because the user gave a feed/input "
            "condition and asked what outputs or product outcomes are possible.\n"
            f"User request: {user_message}"
        )

    return user_message


def _build_missing_inputs_message(classification, plan) -> str:
    if classification.goal == "product_to_feed_sweep":
        return "That sounds like a product-to-feed sweep. What product amount and ABV do you want to target?"
    if classification.goal == "feed_to_product_sweep":
        return "That sounds like a feed-to-product sweep. What feed volume and ABV do you want to start with?"
    if classification.goal == "solve_rayleigh_batch_variables":
        return "That sounds like a direct Rayleigh solve. Give me the missing feed, product, or stopping-composition values for one supported case."
    if classification.goal == "solve_mole_balance":
        return "That sounds like a mole-balance solve. Give me more known feed, product, or bottoms values so I can solve the remaining variables."
    return build_plan_confirmation_message(classification, plan)


def _build_tentative_confirmation_message(classification, plan) -> str:
    if classification.goal == "product_to_feed_sweep":
        return (
            "I think this is a product-to-feed sweep: you have a desired product target and want "
            "possible feed inputs. You gave a target product amount and ABV, so I can convert that "
            "internally and explore feed combinations. Is that the sweep you want me to run?"
        )
    if classification.goal == "feed_to_product_sweep":
        return (
            "I think this is a feed-to-product sweep: you have a starting feed and want possible "
            "product outcomes. I can normalize those inputs if needed and explore the sweep. "
            "Is that the sweep you want me to run?"
        )
    if classification.goal == "solve_rayleigh_batch_variables":
        return (
            "I think this is a direct Rayleigh-constrained solve. You’ve given enough feed, product, "
            "or stopping-composition values that I can normalize them and solve the missing variables "
            "using the Rayleigh equation plus the balances. Is that the calculation you want me to run?"
        )
    if classification.goal == "solve_mole_balance":
        return (
            "I think this is a mole-balance solve. You’ve given enough batch values that I can "
            "normalize them and solve the remaining variables from the total and ethanol balances. "
            "Is that the calculation you want me to run?"
        )
    return (
        f"I think this is a `{classification.goal}` request. "
        "Is that the workflow you want me to run?"
    )


def _build_conversation_context() -> dict:
    recent_messages = st.session_state.messages[-6:]
    recent_user_messages = [m["content"] for m in recent_messages if m["role"] == "user"][-3:]
    recent_assistant_messages = [m["content"] for m in recent_messages if m["role"] == "assistant"][-3:]

    last_execution_result = st.session_state.last_execution_result
    last_result_columns = last_execution_result.columns if last_execution_result is not None else []
    last_workflow_name = last_execution_result.workflow_name if last_execution_result is not None else None

    pending_goal = None
    if st.session_state.pending_classification is not None:
        pending_goal = st.session_state.pending_classification.goal
    elif st.session_state.pending_plan is not None:
        pending_goal = st.session_state.pending_plan.workflow_name

    return {
        "awaiting_confirmation": st.session_state.awaiting_confirmation,
        "has_pending_plan": st.session_state.pending_plan is not None,
        "pending_goal": pending_goal,
        "pending_user_request": st.session_state.pending_user_request,
        "has_last_execution_result": last_execution_result is not None,
        "last_workflow_name": last_workflow_name,
        "last_result_columns": last_result_columns,
        "recent_user_messages": recent_user_messages,
        "recent_assistant_messages": recent_assistant_messages,
        "recent_request_context": st.session_state.recent_request_context,
    }


def build_plan_confirmation_message(classification, plan) -> str:
    if classification.goal == "product_to_feed_sweep":
        if not plan.ready_to_execute:
            return (
                "That sounds like a product-to-feed sweep: you know the desired product and want to "
                "explore feed combinations. I need the target product amount and ABV first."
            )
        return (
            "I understand this as a product-to-feed sweep. You gave a target product amount and ABV, "
            "so I’ll convert that internally to D and xDavg, then explore possible feed volume and "
            "feed ABV combinations. Should I run this sweep?"
        )

    if classification.goal == "solve_rayleigh_batch_variables":
        if not plan.ready_to_execute:
            return (
                "This looks like a direct Rayleigh-constrained solve, but I need the remaining values for a supported case before I can run it."
            )
        return (
            "I understand this as a direct Rayleigh-constrained solve. I’ll normalize any feed, product, "
            "or bottoms inputs into internal variables, then use the Rayleigh equation together with the "
            "total and ethanol balances to solve the missing batch variables. Should I run this calculation?"
        )

    if classification.goal == "solve_mole_balance":
        if not plan.ready_to_execute:
            return (
                "This looks like a mole-balance solve, but I need more known batch values before I can run it."
            )
        return (
            "I understand this as a mole-balance solve. I’ll normalize any feed, product, or bottoms "
            "volume and ABV inputs into internal variables, then use the total and ethanol balances to "
            "solve the requested unknowns. Should I run this calculation?"
        )

    if classification.goal == "feed_to_product_sweep" and plan.ready_to_execute:
        return (
            "I understand this as a feed-to-product sweep. I’ll normalize your starting feed inputs "
            "if needed, sweep xB across feasible stopping compositions, and calculate D and xDavg "
            "outcomes. Should I run this sweep?"
        )

    if not plan.ready_to_execute:
        return (
            f"I understand this as a `{classification.goal}` request, but I need values for "
            f"{', '.join(plan.missing_inputs)} before I can run it."
        )

    key_normalization = plan.normalization_steps[0] if plan.normalization_steps else "No input normalization is needed."
    key_calculation = plan.calculation_steps[0] if plan.calculation_steps else "I have the workflow ready to run."
    return (
        f"I understand this as a `{classification.goal}` request. "
        f"{key_normalization} {key_calculation} "
        "Should I run this sweep?"
    )


def _render_plot(execution_result) -> None:
    if not execution_result.success:
        return
    normalized_inputs = execution_result.normalized_inputs
    try:
        if execution_result.workflow_name == "feed_to_product_sweep":
            if "W0" not in normalized_inputs or "x0" not in normalized_inputs:
                st.warning("Plot inputs are incomplete, so the plot could not be shown.")
                return
            create_plot_D_vs_xDavg_in_L_ABV(
                execution_result.rows,
                float(normalized_inputs["W0"]),
                float(normalized_inputs["x0"]),
            )
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            return

        if execution_result.workflow_name == "product_to_feed_sweep":
            if "D" not in normalized_inputs or "xDavg" not in normalized_inputs:
                st.warning("Plot inputs are incomplete, so the plot could not be shown.")
                return
            create_plot_W0_vs_x0_combinations_in_L_ABV(
                execution_result.rows,
                float(normalized_inputs["D"]),
                float(normalized_inputs["xDavg"]),
            )
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            return
        st.info("No plot is currently available for this workflow.")
    except Exception as exc:
        st.warning(f"Plot display failed: {exc}")


def _handle_confirmation_message(user_message: str) -> None:
    route_name = safe_confirmation_fallback(user_message, True)
    if route_name == "confirm_pending_workflow":
        try:
            result = execute_workflow(
                st.session_state.pending_classification,
                st.session_state.pending_plan,
            )
            st.session_state.last_execution_result = result
            st.session_state.last_classification = st.session_state.pending_classification
            st.session_state.last_plan = st.session_state.pending_plan
            st.session_state.last_user_request = st.session_state.pending_user_request
            st.session_state.awaiting_confirmation = False
            st.session_state.pending_classification = None
            st.session_state.pending_plan = None
            st.session_state.pending_user_request = None
            _clear_pending_intent()
            if result.success:
                add_assistant_message(
                    "Done. I ran the workflow. The results are available in the Results section below.",
                )
            else:
                add_assistant_message(
                    "I ran the planned workflow, but the deterministic engineering layer returned an execution failure. The details are shown below.",
                )
        except Exception as exc:
            st.session_state.awaiting_confirmation = False
            st.session_state.pending_classification = None
            st.session_state.pending_plan = None
            st.session_state.pending_user_request = None
            _clear_pending_intent()
            st.session_state.execution_error = str(exc)
            add_assistant_message("The workflow execution failed. Check the debug details in the Results section.")
        return

    if route_name == "cancel_pending_workflow":
        st.session_state.awaiting_confirmation = False
        st.session_state.pending_classification = None
        st.session_state.pending_plan = None
        st.session_state.pending_user_request = None
        _clear_pending_intent()
        add_assistant_message("Okay. Send a revised request when you want to adjust the plan.")
        return

    add_assistant_message(
        "Please reply yes to run the planned workflow, or no to revise the request.",
    )


def _handle_general_conversation(user_message: str) -> None:
    try:
        response = respond_to_general_message(
            user_message=user_message,
            last_user_request=st.session_state.last_user_request,
            last_classification=st.session_state.last_classification,
            last_plan=st.session_state.last_plan,
            last_execution_result=st.session_state.last_execution_result,
            model_name=st.session_state.model_name,
        )
        add_assistant_message(response)
    except Exception as exc:
        st.session_state.last_error = str(exc)
        add_assistant_message("I hit a conversation error. Check the debug details in the Results section.")


def _handle_explanation_request() -> None:
    if st.session_state.last_execution_result is None:
        add_assistant_message(
            "I do not have a completed result to explain yet. Describe a batch distillation task first.",
        )
        return

    try:
        explanation = explain_execution_result(
            user_message=st.session_state.last_user_request,
            classification=st.session_state.last_classification,
            plan=st.session_state.last_plan,
            execution_result=st.session_state.last_execution_result,
            model_name=st.session_state.model_name,
        )
        add_assistant_message(explanation)
    except Exception as exc:
        st.session_state.result_explanation_error = str(exc)
        add_assistant_message("Result explanation failed. Check the debug details in the Results section.")


def _handle_result_followup_question(user_message: str) -> bool:
    result = st.session_state.last_execution_result
    if result is None:
        return False

    target_abv = extract_abv_target_percent(user_message)
    answer = answer_result_question(user_message, result)
    if answer is not None:
        st.session_state.last_result_query_row = answer.matched_row
        st.session_state.last_result_query_metadata = {
            "matched_column": answer.matched_column,
            "target_value": answer.target_value,
            "returned_columns": answer.returned_columns,
        }
        st.session_state.last_result_query_debug = {
            "routed_as_result_followup": True,
            "detected_abv_target": target_abv,
            "matched_row": answer.matched_row,
            "matched_column": answer.matched_column,
            "returned_columns": answer.returned_columns,
        }
        add_assistant_message(answer.answer)
        return True

    st.session_state.last_result_query_debug = {
        "routed_as_result_followup": True,
        "detected_abv_target": target_abv,
        "matched_row": None,
    }
    add_assistant_message(
        "I can discuss the current results, but I could not find the requested value in the result table.",
    )
    return True


def _handle_matched_row_followup(user_message: str) -> bool:
    if not refers_to_matched_row(user_message):
        return False

    answer = answer_followup_about_matched_row(
        user_message,
        st.session_state.last_result_query_row,
    )
    if answer is None:
        return False

    st.session_state.last_result_query_metadata = {
        "matched_column": answer.matched_column,
        "target_value": answer.target_value,
        "returned_columns": answer.returned_columns,
    }
    st.session_state.last_result_query_debug = {
        "routed_as_matched_row_followup": True,
        "matched_row": answer.matched_row,
        "returned_columns": answer.returned_columns,
    }
    add_assistant_message(answer.answer)
    return True


def _handle_new_task_request(
    classification_message: str,
    user_message: str | None = None,
    classification_hint: str | None = None,
) -> None:
    try:
        classification = classify_goal(
            classification_message,
            model_name=st.session_state.model_name,
            classification_hint=classification_hint,
        )
        plan = create_workflow_plan(classification)
        _clear_pending_intent()
        st.session_state.pending_classification = classification
        st.session_state.pending_plan = plan
        st.session_state.pending_user_request = user_message or classification_message

        if classification.confidence < 0.45:
            st.session_state.awaiting_confirmation = False
            add_assistant_message(_build_missing_inputs_message(classification, plan))
        elif plan.ready_to_execute and classification.confidence >= 0.75:
            st.session_state.awaiting_confirmation = True
            add_assistant_message(build_plan_confirmation_message(classification, plan))
        elif plan.ready_to_execute:
            st.session_state.awaiting_confirmation = True
            add_assistant_message(_build_tentative_confirmation_message(classification, plan))
        else:
            st.session_state.awaiting_confirmation = False
            if classification.goal == "product_to_feed_sweep":
                st.session_state.pending_intent_hint = "product_to_feed_sweep"
                st.session_state.pending_intent_context = st.session_state.pending_user_request
                add_assistant_message(_build_missing_inputs_message(classification, plan))
            elif classification.goal == "feed_to_product_sweep":
                add_assistant_message(_build_missing_inputs_message(classification, plan))
            else:
                add_assistant_message(build_plan_confirmation_message(classification, plan))
    except Exception as exc:
        st.session_state.last_error = str(exc)
        add_assistant_message("Classification failed. Check the debug details in the Results section.")


def _handle_intent_hint_request(user_message: str) -> bool:
    return False


def _handle_pending_intent_completion(user_message: str) -> bool:
    return False


def _handle_workflow_clarification(user_message: str, route) -> None:
    classification_hint = route.likely_goal if route.likely_goal != "unknown" else None
    combined_message = _build_contextual_classification_message(
        user_message,
        likely_goal=classification_hint,
        use_previous_context=True,
    )
    _handle_new_task_request(
        combined_message,
        user_message=user_message,
        classification_hint=classification_hint,
    )


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Describe the batch distillation task you want to run. "
                    "I’ll classify the goal, make a plan, and ask for confirmation before executing."
                ),
            }
        ]
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4o-mini"
    if "pending_classification" not in st.session_state:
        st.session_state.pending_classification = None
    if "pending_plan" not in st.session_state:
        st.session_state.pending_plan = None
    if "pending_user_request" not in st.session_state:
        st.session_state.pending_user_request = None
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if "last_execution_result" not in st.session_state:
        st.session_state.last_execution_result = None
    if "last_classification" not in st.session_state:
        st.session_state.last_classification = None
    if "last_plan" not in st.session_state:
        st.session_state.last_plan = None
    if "last_user_request" not in st.session_state:
        st.session_state.last_user_request = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "execution_error" not in st.session_state:
        st.session_state.execution_error = None
    if "result_explanation_error" not in st.session_state:
        st.session_state.result_explanation_error = None
    if "last_result_query_debug" not in st.session_state:
        st.session_state.last_result_query_debug = None
    if "pending_intent_hint" not in st.session_state:
        st.session_state.pending_intent_hint = None
    if "pending_intent_context" not in st.session_state:
        st.session_state.pending_intent_context = None
    if "recent_request_context" not in st.session_state:
        st.session_state.recent_request_context = None
    if "last_route" not in st.session_state:
        st.session_state.last_route = None
    if "router_error" not in st.session_state:
        st.session_state.router_error = None
    if "last_result_query_row" not in st.session_state:
        st.session_state.last_result_query_row = None
    if "last_result_query_metadata" not in st.session_state:
        st.session_state.last_result_query_metadata = None


st.set_page_config(page_title="Batch Distillation Interface Agent Prototype", layout="wide")
_init_session_state()

st.title("Batch Distillation Interface Agent Prototype")
st.write(
    "This prototype uses a chat-based interface to classify the request, plan a workflow, "
    "ask for confirmation, and run the first supported deterministic workflow."
)
st.caption(
    "The conversation stays in a contained scrollable panel below. Results appear separately underneath and do not intentionally auto-scroll into view."
)

with st.sidebar:
    st.header("Classifier settings")
    st.caption("LLM structured classifier")
    st.session_state.model_name = st.text_input("Model name", value=st.session_state.model_name)

chat_container = st.container(height=420, border=True)
CHAT_CONTAINER = chat_container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_message = st.chat_input("Describe the batch distillation task you want to run...")

if user_message:
    st.session_state.router_error = None
    st.session_state.last_error = None
    st.session_state.execution_error = None
    st.session_state.result_explanation_error = None
    st.session_state.last_result_query_debug = None
    st.session_state.last_route = None
    _update_recent_request_context(user_message)
    add_user_message(user_message)

    if _handle_matched_row_followup(user_message):
        pass
    else:
        conversation_context = _build_conversation_context()
        try:
            route = route_conversation_message(
                user_message=user_message,
                conversation_context=conversation_context,
                model_name=st.session_state.model_name,
            )
            st.session_state.last_route = route
        except Exception as exc:
            st.session_state.router_error = str(exc)
            fallback_route = safe_confirmation_fallback(user_message, st.session_state.awaiting_confirmation)
            if fallback_route == "confirm_pending_workflow" or fallback_route == "cancel_pending_workflow":
                _handle_confirmation_message(user_message)
            else:
                _handle_general_conversation(user_message)
            route = None

        if route is not None:
            if route.route == "confirm_pending_workflow":
                _handle_confirmation_message(user_message)
            elif route.route == "cancel_pending_workflow":
                _handle_confirmation_message("no")
            elif route.route == "new_workflow_request":
                _handle_new_task_request(
                    _build_contextual_classification_message(
                        user_message,
                        likely_goal=route.likely_goal if route.likely_goal != "unknown" else None,
                        use_previous_context=route.uses_previous_context,
                    ),
                    user_message=user_message,
                    classification_hint=route.likely_goal if route.likely_goal != "unknown" else None,
                )
            elif route.route == "workflow_clarification":
                _handle_workflow_clarification(user_message, route)
            elif route.route == "result_question":
                if st.session_state.last_execution_result is not None:
                    _handle_result_followup_question(user_message)
                else:
                    add_assistant_message("I do not have a completed result table to query yet.")
            elif route.route == "result_explanation_request":
                _handle_explanation_request()
            else:
                _handle_general_conversation(user_message)

st.divider()
st.header("Results")

if st.session_state.last_error:
    st.error("LLM classification failed.")
    with st.expander("Debug error details"):
        st.code(st.session_state.last_error)

if st.session_state.router_error:
    st.error("LLM conversation routing failed.")
    with st.expander("Debug error details"):
        st.code(st.session_state.router_error)

if st.session_state.execution_error:
    st.error("Workflow execution failed.")
    with st.expander("Debug error details"):
        st.code(st.session_state.execution_error)

if st.session_state.result_explanation_error:
    st.error("Result explanation failed.")
    with st.expander("Debug error details"):
        st.code(st.session_state.result_explanation_error)

if st.session_state.last_execution_result is None:
    st.info("No workflow has been executed yet.")
else:
    result = st.session_state.last_execution_result
    if result.success:
        st.success(result.message)
    else:
        st.error(result.message)

    if result.warnings:
        for warning in result.warnings:
            st.info(warning)

    with st.expander("Show results table and plot", expanded=True):
        if result.rows:
            st.subheader("Results table")
            st.dataframe(result.rows, use_container_width=True)

            st.subheader("Plot")
            _render_plot(result)
        else:
            st.info("This workflow did not produce table rows.")

    with st.expander("Normalized inputs", expanded=False):
        st.json(result.normalized_inputs)
    with st.expander("Execution parameters", expanded=False):
        st.json(result.execution_parameters)
    with st.expander("Debug: result columns and first row", expanded=False):
        st.write("workflow_name:", result.workflow_name)
        st.write("columns:", result.columns)
        if result.rows:
            st.write("first row keys:", list(result.rows[0].keys()))
            st.json(result.rows[0])
        else:
            st.write("No result rows are available.")
    with st.expander("Debug details", expanded=False):
        if st.session_state.last_classification is not None:
            st.markdown("**Classification**")
            st.json(st.session_state.last_classification.model_dump())
        if st.session_state.last_plan is not None:
            st.markdown("**Workflow plan**")
            st.json(st.session_state.last_plan.model_dump())
        st.markdown("**Execution result**")
        st.json(result.model_dump())

if st.session_state.pending_plan is not None:
    with st.expander("Debug: pending plan", expanded=False):
        pending_payload = {
            "pending_user_request": st.session_state.pending_user_request,
            "awaiting_confirmation": st.session_state.awaiting_confirmation,
            "classification": (
                st.session_state.pending_classification.model_dump()
                if st.session_state.pending_classification is not None
                else None
            ),
            "pending_intent_hint": st.session_state.pending_intent_hint,
            "pending_intent_context": st.session_state.pending_intent_context,
            "recent_request_context": st.session_state.recent_request_context,
            "plan": (
                st.session_state.pending_plan.model_dump()
                if st.session_state.pending_plan is not None
                else None
            ),
        }
        st.json(pending_payload)

if st.session_state.last_result_query_debug is not None:
    with st.expander("Debug: result follow-up routing", expanded=False):
        st.json(st.session_state.last_result_query_debug)

if st.session_state.last_route is not None:
    with st.expander("Debug: conversation route", expanded=False):
        st.json(st.session_state.last_route.model_dump())
