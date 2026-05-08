## Current demo

`OPENAI_API_KEY` must be present in `.env` for the LLM-based ProblemStructurer.

`smoke_test_graph.py` now checks multiple graph paths, including two successful solve paths and one clarification path.

## Setup

1. Run `uv sync`.
2. Create a `.env` file in the project root.
3. Add `OPENAI_API_KEY=...` to `.env`.
4. Make sure OpenAI API access, internet access, and available API quota are present before running the graph-based demos.

`graph_demo.py`, `chat_cli.py`, `streamlit_app.py`, and the current `smoke_test_graph.py` all require live OpenAI access.

## Architecture

The assistant uses a simple four-node LangGraph workflow:

- `Face`: receives the user request.
- `ProblemStructurer`: uses the LLM to turn the request into a structured calculation request.
- `ValidationCalculation`: calls deterministic Python engineering tools.
- `ResultExplainer`: formats the result or clarification response for the user.

## Engineering boundary

The LLM does not perform the batch distillation calculations. Deterministic Python tools perform the Rayleigh and mole-balance calculations, and chemical engineering assumptions are intentionally limited to the current formulas and VLE lookup.

Project structure:

```text
v02/
  graph_demo.py
  smoke_test_graph.py
  chat_cli.py
  streamlit_app.py
  agents/
  engineering/
```

Run with the project virtual environment:

```powershell
.venv\Scripts\python.exe .\graph_demo.py
.venv\Scripts\python.exe .\smoke_test_graph.py
.venv\Scripts\python.exe .\chat_cli.py
.venv\Scripts\streamlit.exe run .\streamlit_app.py
```

If you use `uv`, you can also run:

```powershell
uv run python .\graph_demo.py
uv run python .\smoke_test_graph.py
uv run python .\chat_cli.py
uv run streamlit run .\streamlit_app.py
```
