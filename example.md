# Running the RetroSynth agents

This guide shows how to run the full pipeline, and how to run each agent on its own.

The pipeline has five agents:

| # | Module | Agent | Uses an LLM? |
|---|--------|-------|--------------|
| 1 | `predictor.py` | `SingleStepPredictor`: proposes precursor sets for one molecule | yes (or a custom backend) |
| 2 | `search.py` | `RetroSearchAgent`: expands the tree until every leaf is in stock | only through agent 1 |
| 3 | `ranker.py` | `RouteRanker`: validates and ranks routes | no |
| 4 | `renderer.py` | `RouteRenderer`: draws each route as a PNG | no |
| 5 | `validator.py` | `RouteValidator`: checks each reaction step (gate, then GFN2-xTB ΔG < 0 and ΔH < 0), then an LLM review | yes (for the review only; the energy check runs without it) |

`orchestrator.py` is an alternative to agents 1–2: Claude itself proposes monomers and reaction steps, checks each step with agent 5's gate and energies, and revises the ones that fail. It reaches Claude through the Anthropic API or the local Claude Code CLI. See [Claude orchestrator](#claude-orchestrator).

## 1. Setup

Run everything from the project folder.

```bash
conda create -n retrosynth -c conda-forge python=3.11 rdkit numpy scipy pillow tblite-python pytest
conda activate retrosynth
pip install "dspy==3.4.1" python-dotenv
```

Install dspy with pip only. Installing it with conda as well (`conda install conda-forge::dspy`) puts an older dspy's files over the pip install while `pip show dspy` still reports 3.4.1. Typed answers then come back as plain text, so the predictor's lists break. To repair: `conda remove --force dspy`, `pip uninstall dspy`, then `pip install "dspy==3.4.1"`.

To use the orchestrator's `--claude-code` option, install the Claude Code CLI and log in once by running `claude`. Check that `claude -p "Reply with OK"` prints `OK`.

API keys go in a `.env` file in the project folder. The file is read automatically, and git ignores it:

```
MISTRAL_API_KEY=your-key-here           # used by agent 5 (validator)
ANTHROPIC_API_KEY=your-key-here         # used by agent 1 (predictor) and the orchestrator (backend = api), with the default models
```

The orchestrator with `--claude-code` (`backend = claude_code`) needs no API key: it uses the Claude Code CLI's own login.

A variable already set in your shell takes precedence over `.env`.

You need a building-block file: one SMILES per line, optionally followed by a name. The project does not include one; create `stock.smi` in the project folder, for example:

```
smiles name
OC(=O)c1ccccc1O salicylic_acid
CC(=O)OC(C)=O acetic_anhydride
CC(=O)O acetic_acid
```

Settings for every agent live in `config.ini`. See [section 6](#6-settings-per-agent).

## 2. Run the full pipeline

```bash
python agent.py "CC(=O)Oc1ccccc1C(=O)O" --stock stock.smi
```

This runs agents 1 → 4 on aspirin. The output looks like this:

```
Route 1: score=0.800 steps=1 valid=True
  CC(=O)Oc1ccccc1C(=O)O <- CC(=O)OC(C)=O + O=C(O)c1ccccc1O (p=0.80)
Image: retro_routes/route_01.png
```

Options:

| Flag | Meaning |
|------|---------|
| `smiles` | target molecule (required) |
| `--stock PATH` | building-block file (required) |
| `--context TEXT` | extra instructions for the predictor |
| `--config PATH` | settings file (default: `config.ini`) |

To use a different settings file:

```bash
python agent.py "CC(=O)Oc1ccccc1C(=O)O" --stock stock.smi --config my_settings.ini
```

Agent 5 (validator) checks each route only when `enabled = true` in `[validator]`:

```bash
python agent.py "[*]NCCCCCCNC(=O)CCCCC(=O)[*]" --stock stock.smi
```

## 3. Run one agent at a time

Each example is a short Python script. Run it from the project folder, or set `PYTHONPATH` to the project folder.

Load the settings and the stock once, and reuse them:

```python
from config import load_config
from molecules import load_stock

config = load_config()                 # reads config.ini
stock = load_stock("stock.smi")
```

### Agent 1: single-step predictor

Input: one molecule (SMILES). Output: up to `k` disconnections, each with a list of reactant SMILES and a confidence in [0, 1].

With the LLM (needs `ANTHROPIC_API_KEY`):

```python
from predictor import SingleStepPredictor

predictor = SingleStepPredictor(config.predictor)
for d in predictor("CC(=O)Oc1ccccc1C(=O)O"):
    print(round(d.prob, 2), d.reactants)
```

Without an LLM, pass your own backend. A backend is any function that takes a SMILES string and returns `(reactants, probability)` pairs. The predictor checks each reactant SMILES, drops invalid, trivial, and duplicate entries, and keeps the top `k`:

```python
from predictor import SingleStepPredictor

def my_backend(smiles):
    if smiles == "CC(=O)Oc1ccccc1C(=O)O":
        return [(["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"], 0.8)]
    return []

predictor = SingleStepPredictor(config.predictor, backend=my_backend)
print(predictor("CC(=O)Oc1ccccc1C(=O)O"))
```

### Agent 2: tree search

Input: a target molecule. Output: solved route trees (`TreeNode`), best first. Every leaf is in stock. The search uses agent 1, so give it a predictor:

```python
from predictor import SingleStepPredictor
from search import RetroSearchAgent

predictor = SingleStepPredictor(config.predictor, backend=my_backend)
search = RetroSearchAgent(predictor, stock, config.search)
routes = search("CC(=O)Oc1ccccc1C(=O)O")

for root in routes:
    print(round(root.step_prob, 2), root.smiles, "->", [c.smiles for c in root.children])
```

The search limits come from `[search]` in `config.ini`: `max_depth`, `beam`, and `max_routes`.

### Agent 3: ranker

Input: route trees from agent 2. Output: `RankedRoute` objects, valid routes first, then by score. Each one has `score`, `n_steps`, `valid`, and `issues`. This agent needs no LLM:

```python
from ranker import RouteRanker

ranked = RouteRanker(stock)(routes)
for rr in ranked:
    print(f"score={rr.score:.3f} steps={rr.n_steps} valid={rr.valid}")
    for issue in rr.issues:
        print("  !", issue)
```

Validity checks: every product's heavy atoms must be supplied by its precursors, and every leaf must be in stock.

### Agent 4: renderer

Input: ranked routes. Output: one PNG per route, with one row per reaction step. The target is at the top. Each row shows the precursors (marked "in stock" where they are) and the confidence:

```python
from config import RendererConfig
from renderer import RouteRenderer

renderer = RouteRenderer(RendererConfig(out_dir="my_routes", width=250, height=200))
paths = renderer(ranked)
print(paths)   # [PosixPath('my_routes/route_01.png'), ...]
```

### Agent 5: validator

Input: a route tree from agent 2, the `root` of a `RetroSearchAgent` result. Each internal node is one reaction step, and its children are the reactants. The validator checks the whole reaction path, step by step.

```python
from config import ValidatorConfig
from validator import RouteValidator

validator = RouteValidator(config.validator)
result = validator(routes[0])          # routes from agent 2
print(result.valid)
print(result.rationale)
for s in result.steps:
    print(" + ".join(s.reactants), "->", s.product, s.gate_ok, s.deltas, s.ceiling_temperature)
```

The validator runs in two stages:

1. **Chemistry gate for each step, no LLM.** For every step, reactants → product:
   - the reactant atoms must cover the product, and
   - the leftover atoms must form a known small molecule (H₂O, H₂, NH₃, HCl, CH₃OH, C₂H₅OH, CH₃COOH, CO₂).

   A polymer step (product with `[*]` end points) must also be valid PSMILES with two end points. If any step fails, the route is returned as `valid=False`, with the failing steps named, and no LLM call is made. This stage runs offline. For example, this returns immediately because nitrogen is missing:

   ```python
   from molecules import TreeNode
   bad = TreeNode("[*]CC(N)[*]", 0.9, [TreeNode("C=C")])
   result = validator(bad)
   print(result.valid, result.rationale)    # False, rejected by the chemistry gate: ...
   ```

2. **Thermodynamic check (GFN2-xTB).** Each molecule in a step is built in 3D with RDKit and relaxed with GFN2-xTB through the `tblite` library. The default criteria are the free-energy change ΔG and the enthalpy change ΔH at 298 K and 1 atm, in kcal/mol. H and G are the electronic energy plus thermal corrections from one numerical Hessian per molecule (harmonic vibrations, rigid rotor, ideal gas). The route passes only if every step has every configured change below zero. A step without a value fails. A molecule with an imaginary mode is reported as not a minimum, and the step fails. When both ΔG and ΔH are computed and ΔH, ΔS < 0, each step also reports its ceiling temperature Tc = ΔH/ΔS, above which ΔG turns positive. `criterion` takes a comma-separated list of `dg`, `dh` and `de`. Set `backend = orca` to use ORCA, which supports `de` only.

3. **LLM review.** This runs only if the route passed stages 1 and 2. A ReAct agent reads the whole path, one line per step in forward order, and can call these tools in whatever order it chooses. It can reject a route, but it cannot approve one that failed the energy check:
   - `check_step(product, reactants)`: runs the gate on one step
   - `molecule_energy(smiles)`: relaxed electronic energy of one molecule, in Hartree, from the configured backend
   - `step_energy(product, reactants)`: the configured changes (ΔG, ΔH or ΔE) of one step, in kcal/mol

   This stage needs `MISTRAL_API_KEY` in `.env`. The default model is `mistral/open-mistral-nemo`.

To set the ORCA binary for `backend = orca`, see [section 6](#6-settings-per-agent).

**Caveat:** ΔG is a gas-phase estimate from one GFN2-xTB geometry relaxation with a rigid-rotor, harmonic-oscillator correction. It ignores solvent, conformer sampling, and anharmonicity. The rotational symmetry number is taken as 1, which shifts entropy by up to about 0.6 kcal/mol for symmetric molecules. GFN2 magnitudes differ from experiment: for ethylene + H₂ → ethane, GFN2 gives ΔG ≈ −35 kcal/mol, against an experimental value near −24. Use it as a screen, not as a feasibility cutoff. For a polymer step, the repeat unit is modelled as the H-capped unit minus H₂, so ethylene → polyethylene comes out equal to the hydrogenation of ethylene (GFN2: ΔH ≈ −44 kcal/mol, against an experimental polymerisation enthalpy near −22). Treat polymer ΔH and ceiling temperatures as qualitative.

### Claude orchestrator

`orchestrator.py` puts Claude in charge instead of the fixed search. Claude reads the target, proposes monomers and a forward reaction path, and calls two tools while it works:

- `validate_step(product, reactants)`: the agent 5 chemistry gate, then the energy changes from `[validator]` (ΔG and ΔH with GFN2-xTB by default), with PASS or FAIL and the ceiling temperature
- `in_stock(smiles)`: whether starting materials are in the stock file

When a step fails, Claude revises it and validates again. It returns the path as reaction SMILES, for example `C=C>>[*]CC[*]`. The returned path is then re-checked without the LLM: every step must pass the gate and every configured energy criterion, and every starting material must be in stock. Claude cannot mark a route valid on its own.

```bash
python orchestrator.py "[*]CC[*]" --stock stock.smi
```

This needs `ANTHROPIC_API_KEY` in `.env`. To use your local Claude Code login (for example a Claude subscription) instead of an API key, add `--claude-code`:

```bash
python orchestrator.py "[*]CC[*]" --stock stock.smi --claude-code
```

Each request then runs `claude -p` once, with Claude Code's own tools and MCP servers turned off, so Claude only answers; the orchestrator's tools still run in this process. `ANTHROPIC_API_KEY` is removed from the CLI's environment so the CLI uses its login, not the key. Each call adds about 5 s of CLI start-up time. The backend is a dspy engine (`ClaudeCodeEngine`, passed as `dspy.LM(engine=...)`), so dspy caches answers as it does for the API: an identical request is answered from the cache without running the CLI. Set `backend = claude_code` in `[orchestrator]` to make it the default, and `claude_path` if `claude` is not on your `PATH`.

While it runs, the CLI shows each step as it happens:

```
Target  [*]CC[*]
Claude  anthropic/claude-sonnet-5-5  |  energies  gfn2 (dg, dh)  |  up to 20 tool calls

   2.1s  [1] Claude is thinking...
         Thought: Ethylene undergoes addition polymerisation; validate that step.
         Next: validate_step(product=[*]CC[*], reactants=C=C)
   2.1s  -> validate_step (gate, then energies; this can take a while)
  14.8s  <- PASS: gate ok (atoms balance; by-product (none)); dG=-34.7, dH=-43.6 kcal/mol; ceiling temperature 1462 K  (12.7 s)

  16.0s  [2] Claude is thinking...
         ...
Re-checking Claude's 1 step(s) without the LLM
  step 1: C=C -> *CC*  gate ok  dG=-34.7, dH=-43.6 kcal/mol  Tc=1462 K
  31.5s  Route is VALID
```

Each numbered block is one round of Claude's reasoning: its thought, the tool it chose, and the tool's result with how long it took. After Claude answers, every step is re-checked and the verdict printed, followed by a summary, Claude's rationale, and a PNG of the route. Pass `--quiet` to print only the summary. Colour is used in a terminal and turned off when the output is piped or `NO_COLOR` is set.

From Python, pass `progress=` to receive the same events (`thinking`, `thought`, `tool_start`, `tool_end`, `answering`, `recheck_start`, `recheck_step`, `recheck_done`), or leave it out to run silently:

```python
from orchestrator import RetroOrchestrator

result = RetroOrchestrator(stock, config)("[*]CC[*]", context="prefer step-growth routes")
print(result.valid, result.issues)
for s in result.steps:                 # forward order, starting materials first
    print(" + ".join(s.reactants), "->", s.product, s.deltas, s.ceiling_temperature)
print(result.rationale)
```

To see the orchestrator work without an API key, run the offline demo. Claude's side is scripted, but the gate, the GFN2-xTB energies, the stock check, the live display and the re-check are real:

```bash
python demo_offline.py        # all three scenarios, a few seconds
python demo_offline.py 2      # one scenario
```

1. Polyethylene: a correct proposal passes.
2. Polyacrylonitrile: ethylene fails the gate (no nitrogen), Claude revises to acrylonitrile, and that passes.
3. Polyacrylonitrile: Claude skips the tools and claims ethylene works. The re-check rejects the route.

The demo shows the tools and the safety rule working, not Claude's chemistry. For that, run `orchestrator.py` with an API key.

Settings are in `[orchestrator]`: `model`, `max_tokens`, `max_iters` (tool calls before Claude must answer), `backend` (`api` or `claude_code`), `claude_path` and `timeout` (seconds per CLI call). The energy backend and criteria come from `[validator]`; its `enabled` flag does not matter here.

## 4. Use several agents together

Agents 1–4 in one script, without the CLI:

```python
from agent import RetroSynth
from config import load_config
from molecules import load_stock

config = load_config()
stock = load_stock("stock.smi")
pipeline = RetroSynth(stock, config)         # uses the LLM from [predictor]
result = pipeline("", "CC(=O)Oc1ccccc1C(=O)O")

for rr in result.routes:
    print(rr.score, rr.valid)
print(result.images)
print(result.validations)                    # one ValidationResult per route, when [validator] is enabled
```

To run the pipeline offline, pass a backend as the third argument, as in agent 1: `RetroSynth(stock, config, backend=my_backend)`.

## 5. Run the tests

The tests don't need an API key or ORCA. They use stub predictors and fake energies. The GFN2 tests run real GFN2-xTB calculations when `tblite` is installed, and they are skipped otherwise. The ORCA test is skipped unless ORCA is installed:

```bash
python -m pytest tests -q
```

To run one test file:

```bash
python -m pytest tests/test_validator.py -q
```

## 6. Settings per agent

Each agent has its own section in `config.ini`:

| Section | Keys | Used by |
|---------|------|---------|
| `[predictor]` | `model`, `max_tokens`, `k` | agent 1 |
| `[search]` | `max_depth`, `beam`, `max_routes` | agent 2 |
| `[renderer]` | `out_dir`, `width`, `height` | agent 4 |
| `[validator]` | `enabled`, `backend`, `criterion`, `model`, `max_tokens`, `max_iters`, `orca_path`, `method`, `timeout`, `workdir` | agent 5 |
| `[orchestrator]` | `model`, `max_tokens`, `max_iters`, `backend`, `claude_path`, `timeout` | Claude orchestrator (`orchestrator.py`) |

Notes:

- `model` takes any dspy model string, such as `anthropic/claude-sonnet-5-5` or `openai/gpt-4o`. Set a different model per agent.
- Unknown sections or keys are errors. A typo such as `modle` stops the run, with a message naming the key.
- `enabled` takes `true` or `false`.
- `backend` means different things in two sections:
  - in `[validator]`, the energy source: `gfn2` (default, GFN2-xTB through `tblite`, no external binary) or `orca`;
  - in `[orchestrator]`, how Claude is reached: `api` (default, the Anthropic API with `ANTHROPIC_API_KEY`) or `claude_code` (the local `claude -p` CLI with its own login). `--claude-code` on the command line selects `claude_code` for one run.
- `criterion` is a comma-separated list; every listed change must be below zero. `dg` (free energy) and `dh` (enthalpy) need `gfn2`; `de` (electronic energy) works with either backend. Default `dg, dh`. A condensation step with a small or positive ΔH that is driven by entropy fails `dh`; drop it if you want to accept those.
- `orca_path` is the ORCA binary name on your `PATH`, or its full path, for example `/opt/orca/orca`. It is used only when `backend = orca`.
- `claude_path` is the Claude Code CLI on your `PATH`, or its full path. `timeout` in `[orchestrator]` is the limit in seconds for each CLI call; `timeout` in `[validator]` is the limit for each ORCA run. Both are used only by their own backend (`claude_code`, `orca`).
- `method` is the ORCA keyword line. The default `B3LYP D3BJ def2-SVP Opt TightSCF` runs a geometry optimisation first, then a single point.
- Agents 2, 3, and 4 have no model. Agent 2 uses the predictor's model, and agents 3 and 4 use no LLM.

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `No route to in-stock building blocks found` | Add more building blocks to the stock file, or raise `max_depth` in `[search]`. |
| API key missing, or authentication error | Check the key name and value in `.env` (`MISTRAL_API_KEY` for agent 5). |
| `403` "not available in your subscription tier" | The model needs a higher Mistral tier. Use `mistral/open-mistral-nemo`, or change `model` in `[validator]`. |
| `429 Too Many Requests` | The provider rate-limits you. Wait and retry, or use a smaller `k`/`max_iters`. |
| `ORCA not found at 'orca'` | Set `orca_path` in `[validator]` to the full path of the binary, or use `backend = gfn2`. |
| `tblite` import error | Run `conda install -c conda-forge tblite-python` in the environment. |
| `has an imaginary mode` | The relaxed structure is a saddle point, not a minimum. Relax it further or rebuild it from a different conformer. |
| `ORCA did not terminate normally` | Open the `.out` file in `orca_runs/` (or your `workdir`) and read the ORCA error. |
| `config file not found` | Run from the project folder, or pass `--config` with the path. |
| `unknown key` / `unknown config section` | Check the spelling against the table in section 6. |
| `credit balance is too low to access the Anthropic API` | The API account behind `ANTHROPIC_API_KEY` has no credits. Add credits, or run the orchestrator with `--claude-code` to use your Claude Code login. |
| `Claude Code CLI not found at 'claude'` | Install Claude Code, or set `claude_path` in `[orchestrator]` to the full path of the `claude` binary. |
| `Claude Code CLI returned an error: ... Not logged in` (or another login message) | Run `claude` once in a terminal and log in, then check `claude -p "Reply with OK"`. |
| `Claude Code CLI timed out after ... s` | Raise `timeout` in `[orchestrator]`. Each call is retried once, so a stuck call can take up to twice the timeout. |
| `Claude Code CLI failed (exit ...)` | The message shows the CLI's own error. An "unknown option" message means the installed Claude Code is too old for the flags used; update it. |
| Predictor or orchestrator answers arrive as text, not lists, or `zip` errors in the predictor | dspy is a mixed install. See the dspy note in [section 1](#1-setup). |
