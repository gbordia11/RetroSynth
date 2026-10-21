"""Agent 2: expands the retrosynthetic tree until every leaf is in stock."""

import itertools

import dspy

from config import SearchConfig
from molecules import TreeNode, canonical, route_score


class RetroSearchAgent(dspy.Module):
    """Agent 2: expands the retrosynthetic tree until every leaf is in stock."""

    def __init__(self, predictor, stock, config=None):
        super().__init__()
        self.predictor = predictor
        self.stock = stock
        self.config = config or SearchConfig()
        self._memo = {}

    def forward(self, smiles, context=""):
        self._memo = {}
        target = canonical(smiles)
        if target is None or target in self.stock:
            return []
        routes = self._solve(target, 0, frozenset(), context)
        return routes[: self.config.max_routes]

    def _solve(self, smiles, depth, path, context):
        """Return up to `beam` solved subtrees for `smiles`, best first."""
        if smiles in self.stock:
            return [TreeNode(smiles)]
        if depth >= self.config.max_depth or smiles in path:
            return []
        if (smiles, depth) in self._memo:
            return self._memo[(smiles, depth)]

        solutions = []
        for disc in self.predictor(smiles, context=context):
            child_options = [
                self._solve(r, depth + 1, path | {smiles}, context) for r in disc.reactants
            ]
            if not all(child_options):
                continue
            for combo in itertools.product(*child_options):
                solutions.append(TreeNode(smiles, disc.prob, list(combo)))

        solutions.sort(key=route_score, reverse=True)
        solutions = solutions[: self.config.beam]
        self._memo[(smiles, depth)] = solutions
        return solutions
