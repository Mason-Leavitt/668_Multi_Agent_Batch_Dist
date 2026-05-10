## Interface Agent Prototype

This prototype adds an interface agent for the batch distillation project.
The LLM structured classifier is the primary and only final classifier for natural-language requests.
Deterministic keyword classification is not used as a fallback because it can misclassify natural user phrasing.
Lightweight deterministic feature extraction may still be used only as hint context for the LLM.
`variable_assignments` may contain strings or simple numeric values.

It does not implement tool calling, memory, or multi-agent orchestration yet.

Classification identifies the user's goal.
Workflow planning maps that goal to required inputs, normalization needs, future calculation steps, and result-formatting expectations.
The workflow planner distinguishes between variables mentioned by the user and variables with actual assigned values. A workflow is only `ready_to_execute` when the required values are present.
The app now runs selected deterministic engineering workflows after planning and confirmation.

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

The main UI now focuses on results instead of raw intermediate JSON.
Classification, planning, and raw execution details are available in collapsible debug sections.
The app now uses a chat-based interface for task requests, confirmation, and follow-up explanations.
The app now uses an LLM semantic router before goal classification.
The router decides whether a user message is a new workflow request, a workflow clarification, a confirmation, a result question, a result explanation, or general conversation.
The goal classifier only runs for workflow requests and workflow clarifications.
Assistant chat replies now stream into the chat UI.
Streaming applies to conversational text only; tables, plots, and debug JSON render normally.
The chat history is displayed in a contained scrollable panel.
Results appear below the chat in a separate section.
Generating results does not intentionally auto-scroll the user to the Results section.
The chat now has a general conversation path.
The router now prioritizes actionable workflow requests before general conversation.
The app does not classify every user message as a new workflow, but it does recover multi-turn workflow setup when the user first describes the intent and later supplies values.
New calculation and design requests are classified and planned.
Lower-confidence workflow requests can still move forward through a confirmation step instead of dropping straight into general conversation.
Corrections, explanations, and discussion about results are handled conversationally.
`docs/agent_background.md` provides stable context for explanations and general conversation.
The LLM still does not perform deterministic calculations.
Numeric execution remains deterministic.
Volume and ABV are first-class user inputs, so users do not need to convert requests into moles or mole fractions before starting a supported sweep.
Recent volume and ABV targets can be reused when the user clarifies the intended workflow in the next turn.

If the LLM call fails, the app reports the error instead of pretending to classify the request.
The app can classify intent, plan supported workflows, execute implemented deterministic workflows, display results, explain results, and answer supported result follow-up questions.

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

`feed_to_product_sweep`, `product_to_feed_sweep`, `solve_mole_balance`, and `solve_rayleigh_batch_variables` are executable now.
Other goals can still be classified and planned, but they are not executed yet.
These workflows use deterministic engineering functions from the existing engineering package.
The LLM is used only for classification, not for math.
The agent asks for confirmation in chat before executing deterministic calculations.
The results table and plot appear in a separate Results section below the chat.
Successful workflow runs are also explained automatically in chat.
You can ask follow-up questions like `explain the results` in the chat.
The chat can also answer follow-up questions about the most recent result table.
The first supported result lookup is: given a desired `xDavg` ABV percent, return the nearest `D_volume_L` from the current sweep.
For `product_to_feed_sweep`, the app can also query feed volume by feed/input ABV using nearest-row matching on `x0_abv_percent` and returning `W0_volume_L`.
The app remembers the last matched row for simple follow-ups like `what is the feed volume for that row?`
This lookup uses nearest-row matching, not interpolation.
No interpolation is performed yet.
If a product-to-feed lookup cannot run because the current result table is missing a feed ABV column or feed-volume column, the app reports which column type is missing.
New workflow requests are still classified normally.
Classification, planning, and debug JSON remain available in collapsible expanders.
The LLM is used for classification and explanation only, not calculation.
Plots are shown when a workflow has plotting support.
LLM explanations summarize deterministic results only.
The LLM does not perform calculations.
Validation is handled by the deterministic engineering functions.
The executor does not duplicate engineering validation rules.
Engineering validation failures are caught and returned as clean execution results.
Normalized inputs are shown so users can see what values were passed to the calculations.
`xDavg` is currently reported as ethanol mole fraction.

Test prompts:

1. `Given W0 = 100 mol and x0 = 0.05, what D and xDavg can I get over different xB values?`
2. `If W0 = 100 L at 5% ABV ethanol, what combinations of xDavg and D could I get?`
3. `I want 1 gallon of product at 60% ABV. What feed do I need?`
4. `Given D = 20 mol and xDavg = 0.4, what W0 and x0 combinations could work?`
5. `What starting volume and ABV would I need to get 2 L at 50% ABV?`

`product_to_feed_sweep` accepts product targets in model units or user-friendly volume/ABV units.
It returns possible feed/start conditions.
Result rows include internal model columns and user-friendly volume/ABV columns where available.
The LLM still only classifies and explains; deterministic engineering functions do calculations.
`solve_mole_balance` solves selected batch variables from the total balance and ethanol balance.
It supports model units and user-facing volume/ABV pairs for feed, product, and bottoms values.
It does not enforce Rayleigh behavior and returns one-row result tables.
`solve_rayleigh_batch_variables` solves direct Rayleigh-constrained batch cases.
It currently supports these known-variable sets:
1. `W0, x0, xB`
2. `W0, x0, D`
3. `D, xDavg, x0`
4. `D, xDavg, xB`
It supports model units and user-facing volume/ABV inputs where possible.
It returns one-row result tables and does not currently generate a plot.

## Troubleshooting

- Check that `OPENAI_API_KEY` exists in `.env`.
- Check that dependencies are installed with `uv sync`.
- If structured-output schema warnings appear, confirm `langchain-openai` is using `method="function_calling"` for structured output.

Additional `solve_mole_balance` examples:

1. `I started with 100 L at 10% ABV and collected 10 L at 45% ABV. What is left in the still?`
2. `If my feed is 100 L at 10% ABV and my bottoms are 80 L at 3% ABV, how much product did I collect and what was its ABV?`
3. `Given D = 20 mol, xDavg = 0.4, B = 80 mol, and xB = 0.02, what W0 and x0 did I start with?`

`solve_rayleigh_batch_variables` examples:

1. `Given W0 = 100 mol, x0 = 0.05, and xB = 0.01, calculate D and xDavg.`
2. `Given W0 = 100 mol, x0 = 0.05, and D = 20 mol, calculate xDavg.`
3. `If I want 10 L at 45% ABV and my feed is 10% ABV, how much feed do I need?`
4. `If I want 10 L at 45% ABV and stop the boiler at 2% ABV, what feed do I need?`
