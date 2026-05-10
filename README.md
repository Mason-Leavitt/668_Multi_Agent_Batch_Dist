## Interface Agent Prototype

This prototype adds an interface agent for the batch distillation project.
The LLM structured classifier is the primary and only final classifier for natural-language requests.
Deterministic keyword classification is not used as a fallback because it can misclassify natural user phrasing.
Lightweight deterministic feature extraction may still be used only as hint context for the LLM.
`variable_assignments` may contain strings or simple numeric values.

It does not run engineering calculations yet.
It does not implement tool calling, memory, or multi-agent orchestration yet.

Classification identifies the user's goal.
Workflow planning maps that goal to required inputs, normalization needs, future calculation steps, and result-formatting expectations.
The workflow planner distinguishes between variables mentioned by the user and variables with actual assigned values. A workflow is only `ready_to_execute` when the required values are present.
The app still does not run calculations yet.

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

Install dependencies with `uv`:

```bash
uv sync
```

Run the prototype app with:

```bash
uv run streamlit run app/streamlit_app.py
```

If the LLM call fails, the app reports the error instead of pretending to classify the request.
The agent only classifies intent right now.
Calculations and tool routing will be added later.

## Workflow Planning

After classification, the app creates a deterministic `WorkflowPlan`.
This plan answers:
- which workflow should run
- which inputs are required
- which inputs are already available
- which inputs are still missing
- which normalization steps will be needed
- which calculation steps are expected later
- which result-formatting steps are expected
- whether the request is ready to execute

The next implementation step is to connect one workflow path to the deterministic engineering functions without changing the classifier or planner roles.

## Troubleshooting

- Check that `OPENAI_API_KEY` exists in `.env`.
- Check that dependencies are installed with `uv sync`.
- If structured-output schema warnings appear, confirm `langchain-openai` is using `method="function_calling"` for structured output.
