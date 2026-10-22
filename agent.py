"""Template-free retrosynthesis pipeline driven by a target SMILES.

The five agents live in their own modules:
1. predictor.py - SingleStepPredictor: P_ss(target) -> top-K (precursor set, probability).
2. search.py    - RetroSearchAgent: expands the tree until every leaf is in stock.
3. ranker.py    - RouteRanker: validates each route and ranks the valid ones.
4. renderer.py  - RouteRenderer: draws each route with RDKit as a PNG.
5. validator.py - RouteValidator: chemistry gate, dG/dH check and LLM review (when enabled).

Settings (models, search limits, output folder) come from config.ini.
This module wires the agents together and provides the command-line entry point.

Usage:
    python agent.py "CC(=O)Oc1ccccc1C(=O)O" --stock stock.smi [--config config.ini]
"""

import argparse

import dspy

from config import DEFAULT_CONFIG_PATH, Config, load_config
from molecules import load_stock
from predictor import SingleStepPredictor
from ranker import RouteRanker
from renderer import RouteRenderer
from search import RetroSearchAgent
from validator import RouteValidator, format_deltas


class RetroSynth(dspy.Module):
    """Runs the agents on one target molecule; the validator only when enabled in the config."""

    def __init__(self, stock, config=None, backend=None):
        super().__init__()
        self.config = config or Config()
        self.predictor = SingleStepPredictor(self.config.predictor, backend=backend)
        self.search = RetroSearchAgent(self.predictor, stock, self.config.search)
        self.ranker = RouteRanker(stock)
        self.renderer = RouteRenderer(self.config.renderer)
        self.validator = RouteValidator(self.config.validator) if self.config.validator.enabled else None

    def forward(self, prompt: str, smile: str):
        roots = self.search(smile, context=prompt)
        ranked = self.ranker(roots)
        images = self.renderer(ranked)
        validations = [self.validator(rr.root) for rr in ranked] if self.validator is not None else []
        return dspy.Prediction(routes=ranked, images=images, validations=validations)


def format_route(root):
    """Human-readable lines: one reaction step per line."""
    return [
        f"  {step.smiles} <- {' + '.join(c.smiles for c in step.children)} (p={step.step_prob:.2f})"
        for step in root.steps()
    ]


def main():
    parser = argparse.ArgumentParser(description="Template-free retrosynthesis agents")
    parser.add_argument("smiles", help="target molecule SMILES")
    parser.add_argument("--stock", required=True, help="building blocks file, one SMILES per line")
    parser.add_argument("--context", default="", help="extra instructions for the predictor")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="settings file (INI)")
    args = parser.parse_args()

    config = load_config(args.config)
    stock = load_stock(args.stock)
    pipeline = RetroSynth(stock, config)
    result = pipeline(args.context, args.smiles)

    if not result.routes:
        print("No route to in-stock building blocks found.")
    for i, rr in enumerate(result.routes, 1):
        print(f"Route {i}: score={rr.score:.3f} steps={rr.n_steps} valid={rr.valid}")
        print("\n".join(format_route(rr.root)))
        for issue in rr.issues:
            print(f"  ! {issue}")
    for path in result.images:
        print(f"Image: {path}")
    for i, v in enumerate(result.validations, 1):
        print(f"Validation route {i}: valid={v.valid} favourable={v.favourable}")
        for s in v.steps:
            gate = "ok" if s.gate_ok else "FAIL"
            energy = f" {format_deltas(s.deltas)}" if s.deltas else ""
            tc = f" Tc={s.ceiling_temperature:.0f} K" if s.ceiling_temperature is not None else ""
            print(f"  {' + '.join(s.reactants)} -> {s.product}: gate={gate}{energy}{tc}")
        print(f"  {v.rationale}")


if __name__ == "__main__":
    main()
