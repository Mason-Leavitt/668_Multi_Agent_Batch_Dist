from agents.error_handling import normalize_error_for_user
from agents.graph import build_graph
from agents.session_commands import apply_session_command, parse_session_command


def main() -> None:
    app = build_graph()
    session_knowns: dict[str, float] = {}
    awaiting_clarification = False
    clarification_question: str | None = None
    active_experiment: dict | None = None
    experiment_results: list[dict] | None = None
    experiment_sampled_variable: str | None = None
    experiment_knowns: dict[str, float] | None = None
    experiment_status: str | None = None
    pending_commit_variable: str | None = None
    pending_commit_value: float | None = None
    pending_commit_source: dict | None = None

    print("Batch distillation assistant")
    print("Type a question, or type exit, quit, or q to stop.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not user_message:
            continue

        try:
            session_command = parse_session_command(user_message)
            if session_command["is_session_command"]:
                session_update = apply_session_command(
                    command=session_command,
                    prior_knowns=session_knowns,
                    prior_needs_clarification=awaiting_clarification,
                    prior_clarification_question=clarification_question,
                    active_experiment=active_experiment,
                    experiment_results=experiment_results,
                    experiment_sampled_variable=experiment_sampled_variable,
                    experiment_knowns=experiment_knowns,
                    experiment_status=experiment_status,
                    pending_commit_variable=pending_commit_variable,
                    pending_commit_value=pending_commit_value,
                    pending_commit_source=pending_commit_source,
                )
                session_knowns = session_update["prior_knowns"]
                awaiting_clarification = session_update["prior_needs_clarification"]
                clarification_question = session_update["prior_clarification_question"]
                active_experiment = session_update["active_experiment"]
                experiment_results = session_update["experiment_results"]
                experiment_sampled_variable = session_update["experiment_sampled_variable"]
                experiment_knowns = session_update["experiment_knowns"]
                experiment_status = session_update["experiment_status"]
                pending_commit_variable = session_update["pending_commit_variable"]
                pending_commit_value = session_update["pending_commit_value"]
                pending_commit_source = session_update["pending_commit_source"]
                print(f"\nAssistant: {session_update['message']}")
                continue

            graph_input = {"user_message": user_message}
            if session_knowns or awaiting_clarification or clarification_question:
                graph_input["prior_knowns"] = session_knowns
                graph_input["prior_needs_clarification"] = awaiting_clarification
                graph_input["prior_clarification_question"] = clarification_question
            if active_experiment or experiment_results or experiment_knowns or experiment_sampled_variable:
                graph_input["active_experiment"] = active_experiment
                graph_input["experiment_results"] = experiment_results
                graph_input["experiment_sampled_variable"] = experiment_sampled_variable
                graph_input["experiment_knowns"] = experiment_knowns
                graph_input["experiment_status"] = experiment_status
            if pending_commit_variable is not None or pending_commit_value is not None:
                graph_input["pending_commit_variable"] = pending_commit_variable
                graph_input["pending_commit_value"] = pending_commit_value
                graph_input["pending_commit_source"] = pending_commit_source

            final_state = app.invoke(graph_input)
            session_knowns = final_state.get("knowns", {})
            awaiting_clarification = final_state.get("needs_clarification", False)
            clarification_question = final_state.get("clarification_question")
            active_experiment = final_state.get("active_experiment")
            experiment_results = final_state.get("experiment_results")
            experiment_sampled_variable = final_state.get("experiment_sampled_variable")
            experiment_knowns = final_state.get("experiment_knowns")
            experiment_status = final_state.get("experiment_status")
            pending_commit_variable = final_state.get("pending_commit_variable")
            pending_commit_value = final_state.get("pending_commit_value")
            pending_commit_source = final_state.get("pending_commit_source")
            has_active_experiment_context = bool(
                active_experiment
                or experiment_results
                or experiment_knowns
                or experiment_sampled_variable
            )

            if (
                not awaiting_clarification
                and not has_active_experiment_context
                and final_state.get("intent_type")
                not in {
                    "design_prototyping",
                    "open_ended_guidance",
                    "conceptual_question",
                }
            ):
                session_knowns = {}
                clarification_question = None
                active_experiment = None
                experiment_results = None
                experiment_sampled_variable = None
                experiment_knowns = None
                experiment_status = None
                pending_commit_variable = None
                pending_commit_value = None
                pending_commit_source = None

            if final_state.get("calculation_success") is True:
                active_experiment = None
                experiment_results = None
                experiment_sampled_variable = None
                experiment_knowns = None
                experiment_status = None
                pending_commit_variable = None
                pending_commit_value = None
                pending_commit_source = None

            print(f"\nAssistant: {final_state['final_answer']}")
        except Exception as exc:
            print(f"\nAssistant: {normalize_error_for_user(exc)}")


if __name__ == "__main__":
    main()
