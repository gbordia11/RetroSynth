"""Offline demo of the Claude orchestrator: real GFN2-xTB energies, scripted Claude.

No API key is needed. Claude's side of each conversation (its thoughts, tool choices and final
answer) is scripted with dspy's DummyLM. Everything else is real: the chemistry gate, the GFN2-xTB
dG and dH, the stock check, the live progress display, and the re-check of the returned route.

This shows the tools, the loop and the safety rule working. It does not show Claude's chemistry:
for that, run orchestrator.py with an API key.

Scenarios:
1. polyethylene: Claude proposes ethylene, the step passes, the route is valid.
2. polyacrylonitrile: Claude first proposes ethylene, the gate rejects it (no nitrogen),
   Claude revises to acrylonitrile, the step passes.
3. polyacrylonitrile again: Claude skips the tools and returns the wrong step anyway.
   The re-check rejects the route, so Claude cannot make it valid on its own.

Usage:
    python demo_offline.py            # all three scenarios
    python demo_offline.py 2          # one scenario
"""

import argparse
from dataclasses import replace

import dspy

from config import load_config
from gfn2 import Gfn2Runner
from molecules import canonical
from orchestrator import ProgressPrinter, RetroOrchestrator, format_step

PE = "[*]CC[*]"
PAN = "[*]CC([*])C#N"
STOCK = {canonical(s) for s in ("C=C", "C=CC#N", "C=Cc1ccccc1")}  # ethylene, acrylonitrile, styrene


def think(thought, tool, **args):
    """One scripted ReAct round: Claude's thought and the tool it calls."""
    return {"next_thought": thought, "next_tool_name": tool, "next_tool_args": args}


def answer(steps, rationale):
    """Claude's final answer, after it calls finish."""
    return {"reasoning": rationale, "steps": steps, "rationale": rationale}


SCENARIOS = {
    1: {
        "title": "Polyethylene: a correct proposal passes",
        "target": PE,
        "script": [
            think("Polyethylene is made by addition polymerisation of ethylene. Validate that step.",
                  "validate_step", product=PE, reactants="C=C"),
            think("The step passed. Check that ethylene is purchasable.", "in_stock", smiles="C=C"),
            think("Ethylene is in stock and the only step passed, so the route is complete.", "finish"),
            answer(["C=C>>[*]CC[*]"], "Addition polymerisation of ethylene; the step passed the gate and the energy check."),
        ],
    },
    2: {
        "title": "Polyacrylonitrile: a wrong proposal fails, Claude revises it",
        "target": PAN,
        "script": [
            think("Try ethylene as the monomer for this vinyl polymer.",
                  "validate_step", product=PAN, reactants="C=C"),
            think("The gate failed: the repeat unit has a nitrile, so the monomer needs C and N. "
                  "Acrylonitrile, C=CC#N, supplies both. Validate that instead.",
                  "validate_step", product=PAN, reactants="C=CC#N"),
            think("Acrylonitrile passes. Check that it is purchasable.", "in_stock", smiles="C=CC#N"),
            think("Acrylonitrile is in stock and the step passed.", "finish"),
            answer(["C=CC#N>>[*]CC([*])C#N"], "Ethylene lacks the nitrile nitrogen; acrylonitrile supplies it and the step passed."),
        ],
    },
    3: {
        "title": "Polyacrylonitrile: Claude claims a wrong route, the re-check rejects it",
        "target": PAN,
        "script": [
            think("I am confident ethylene works here; no need to validate.", "finish"),
            answer(["C=C>>[*]CC([*])C#N"], "Ethylene polymerises to the target."),
        ],
    },
}


def run(number, config):
    scenario = SCENARIOS[number]
    print()
    print("=" * 100)
    print(f"Scenario {number}: {scenario['title']}")
    print("Claude is SCRIPTED (offline demo); the gate, GFN2-xTB energies and stock check are real.")
    print("=" * 100)
    printer = ProgressPrinter()
    orch = RetroOrchestrator(STOCK, config, progress=printer)
    orch.lm = dspy.utils.DummyLM(scenario["script"])
    printer.header(scenario["target"], replace(config, orchestrator=replace(config.orchestrator, model="scripted")))
    result = orch(scenario["target"])
    print()
    print(f"Result: valid={result.valid}")
    for i, s in enumerate(result.steps, 1):
        print("  " + format_step(i, s))
    for issue in result.issues:
        print(f"  ! {issue}")
    print(f"Claude's rationale: {result.rationale}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline demo of the Claude orchestrator (scripted Claude, real energies)")
    parser.add_argument("scenarios", nargs="*", type=int, choices=sorted(SCENARIOS), help="scenario numbers (default: all)")
    args = parser.parse_args()
    if not Gfn2Runner.available():
        raise SystemExit("this demo needs tblite for the GFN2-xTB energies: conda install -c conda-forge tblite-python")
    config = load_config()
    config = replace(config, validator=replace(config.validator, backend="gfn2"))
    for number in args.scenarios or sorted(SCENARIOS):
        run(number, config)


if __name__ == "__main__":
    main()
