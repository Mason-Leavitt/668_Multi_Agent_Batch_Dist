## Current demo

`OPENAI_API_KEY` must be present in `.env` for the LLM-based ProblemStructurer.

`smoke_test_graph.py` is an LLM-integration smoke test that checks multiple graph paths, including successful solve paths and a clarification path.

Normal responses are concise by default. If you want a fuller explanation, you can ask with phrases like `explain`, `I don't understand`, or `walk me through it`.

## Setup

1. Run `uv sync`.
2. Create a `.env` file in the project root.
3. Add `OPENAI_API_KEY=...` to `.env`.
4. Make sure OpenAI API access, internet access, and available API quota are present before running the graph-based demos.

Use the `uv`-managed environment when running the project. For example:

```powershell
.venv\Scripts\python.exe .\graph_demo.py
uv run python .\graph_demo.py
```

Do not rely on an arbitrary system Python unless the same dependencies are installed there.

`graph_demo.py`, `chat_cli.py`, `streamlit_app.py`, and the current `smoke_test_graph.py` all require live OpenAI access.

`smoke_test_graph.py` prints short pass/fail-style output so it is easier to scan during demo verification.

## Architecture

The assistant uses a LangGraph workflow with a shared deterministic calculation core:

- `Face`: receives the user request.
- `ProblemStructurer`: uses the LLM to turn the request into a structured calculation request.
- Then the graph routes to one of:
  - `ValidationCalculation` for ready deterministic calculations
  - `DesignAdvisor` for partial-knowns, underdetermined design requests, and illustrative scenario guidance
  - `GuidanceResponder` for broad orientation, conceptual help, and unsupported requests
- `ResultExplainer`: formats deterministic calculation results for the user.

`GuidanceResponder` answers broad "what can you do?" or "what does this variable mean?" questions.
`DesignAdvisor` helps the user move from partial knowns or vague design goals toward one of the supported deterministic workflows.

At a high level, the conversational graph is:

`Face -> ProblemStructurer -> route to ValidationCalculation, DesignAdvisor, or GuidanceResponder -> ResultExplainer when a deterministic calculation runs`

## Engineering boundary

The LLM does not perform the batch distillation calculations. Deterministic Python tools perform the Rayleigh and mole-balance calculations, and chemical engineering assumptions are intentionally limited to the current formulas and VLE lookup.

The assistant also tries to return user-safe error messages. When a deterministic calculation cannot run yet, it will usually ask for more information or explain the engineering validation issue in user-facing language instead of showing raw internal errors.

## Scope

This project is a batch-distillation assistant for simplified Rayleigh and mole-balance calculations.

- It helps structure simplified calculation requests and route them to deterministic Python tools.
- The LLM structures the conversation, while deterministic Python tools perform the calculations.
- Illustrative scenarios are examples to help compare design choices and choose a design basis.

## Session commands

In the CLI and Streamlit chat, you can manage remembered values in the current session with simple commands:

- `start over` or `reset`
- `forget x0`
- `change x0 to 0.08`
- `set W0 to 1500`

These commands affect remembered values in the current CLI or Streamlit session only. Variable names are handled case-insensitively where practical, so commands like `set w0 to 1500` also work.

## Supported workflow examples

`solve_D_given_W0_x0_xDavg`

- Example prompt:
  `"I have 1000 mol of ethanol-water at 5 mol% ethanol. I want the average distillate to be 20 mol% ethanol. How much distillate can I collect?"`

`solve_batch_given_W0_x0_xB`

- Example prompt:
  `"I start with 1000 mol of ethanol-water at 5 mol% ethanol and distill until the still is 1 mol% ethanol. How much distillate do I collect?"`

`check_batch_consistency`

- Example prompt:
  `"Check whether this batch result is consistent: W0=1000 mol, B=763.986 mol, D=236.014 mol, x0=0.05, xB=0.003661, and xDavg=0.20."`
- This path checks total mole balance, ethanol component balance, and Rayleigh consistency.

Open-ended starting request

- Example prompt:
  `"I don't know where to start but I want to conduct a distillation."`
- The assistant can now guide a vague starting request toward one of the supported workflows instead of forcing an immediate calculation.

Partial design request

- Example prompt:
  `"I have 1000 mol at 5 mol% ethanol, help me choose targets."`
- The assistant can compare the known inputs against the supported workflows, explain what is still missing, identify which missing variable could be sampled next, and run compact illustrative scenario results when an existing deterministic solver can support them.
- After scenario examples, you can continue with explicit commands like `use option 2`, `try xB = 0.007`, `show higher xB values`, or `done with this experiment`.
- Natural follow-ups also work for active experiments, for example `choose the second one`, `what if xB is 0.007?`, `vary feed composition instead`, or `explain option 2`.

Underdetermined design request

- Example prompt:
  `"I want to produce about 50 moles of ethanol-water mixture distillate at a 0.2 ethanol mole fraction. How do I set up the still?"`
- The assistant can summarize the provided targets, explain that the request is still underdetermined, and ask for a useful next design basis such as `x0` or `xB`.

Design prototyping request

- Example prompt:
  `"I want a distillate of 50 moles at a 0.2 mole fraction of ethanol. How much initial mole mixture do I need and at what mole fraction?"`
- The assistant can compare the user's knowns against the supported workflows and explain when an additional design basis is still required before a reliable calculation can be completed.

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
