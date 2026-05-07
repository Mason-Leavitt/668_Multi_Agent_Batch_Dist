## Current demo

`OPENAI_API_KEY` must be present in `.env` for the LLM-based ProblemStructurer.

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
