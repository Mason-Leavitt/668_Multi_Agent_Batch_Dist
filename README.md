## Interface Agent Prototype

This prototype adds an interface agent for the batch distillation project.
The primary path now uses an LLM with structured output to classify a user's natural-language request into the `GoalClassification` schema. A deterministic classifier remains available only as a fallback and debug baseline.

It does not run engineering calculations yet.
It does not implement tool calling, memory, or multi-agent orchestration yet.

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

The LLM classifier is now the primary goal classifier.
The deterministic classifier is only a fallback/debug baseline.
The agent only classifies intent right now.
Calculations and tool routing will be added later.
