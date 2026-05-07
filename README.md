## Current demo

`OPENAI_API_KEY` must be present in `.env` for the LLM-based ProblemStructurer.

`smoke_test_graph.py` now checks multiple graph paths, including two successful solve paths and one clarification path.

Project structure:

```text
v02/
  graph_demo.py
  smoke_test_graph.py
  agents/
  engineering/
```

Run with the project virtual environment:

```powershell
.venv\Scripts\python.exe .\graph_demo.py
.venv\Scripts\python.exe .\smoke_test_graph.py
```

If you use `uv`, you can also run:

```powershell
uv run python .\graph_demo.py
uv run python .\smoke_test_graph.py
```
