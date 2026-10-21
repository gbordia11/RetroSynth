"""Agent 3: validates each retrosynthesis route and ranks them."""

from collections import Counter
from dataclasses import dataclass

import dspy

from molecules import element_counts, route_score


@dataclass
class RankedRoute:
    root: object  # TreeNode
    score: float
    n_steps: int
    valid: bool
    issues: list


class RouteRanker(dspy.Module):
    """Agent 3: checks every route for chemical sanity and ranks them."""

    def __init__(self, stock):
        super().__init__()
        self.stock = stock

    def forward(self, routes):
        ranked = [self._check(root) for root in routes]
        ranked.sort(key=lambda r: (not r.valid, -r.score, r.n_steps))
        return ranked

    def _check(self, root):
        issues = []
        for step in root.steps():
            # Necessary condition for a balanced step: the precursors together must
            # supply every heavy atom of the product.
            supplied = Counter()
            for child in step.children:
                supplied += element_counts(child.smiles)
            missing = element_counts(step.smiles) - supplied
            if missing:
                issues.append(f"{step.smiles}: precursors lack {dict(missing)}")
        for leaf in root.leaves():
            if leaf.smiles not in self.stock:
                issues.append(f"leaf not in stock: {leaf.smiles}")
        return RankedRoute(
            root=root,
            score=route_score(root),
            n_steps=sum(1 for _ in root.steps()),
            valid=not issues,
            issues=issues,
        )
