## Interface Agent Prototype

This prototype adds a small deterministic interface agent for the batch distillation project.
It classifies a user's request into a structured goal schema, detects likely known inputs and requested outputs, and records whether input or output conversion will be needed later.

It does not run engineering calculations yet.
It does not call OpenAI.
It does not use LangChain yet.
It does not implement tool calling, memory, or multi-agent orchestration yet.

Install dependencies with `uv`:

```bash
uv sync
```

Run the prototype app with:

```bash
uv run streamlit run app/streamlit_app.py
```

LangChain and OpenAI integration will be added later after the deterministic classifier and schema are stable.
