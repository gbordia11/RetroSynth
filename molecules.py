"""Shared chemistry helpers and the retrosynthesis tree data structures."""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")  # invalid SMILES from the predictor are expected and filtered


def canonical(smiles):
    """Return canonical SMILES, or None if RDKit cannot parse the string."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol else None


def load_stock(path):
    """Read building blocks: one SMILES per line, first field is used ("SMILES name" also works)."""
    stock = set()
    for line in Path(path).read_text().splitlines():
        fields = re.split(r"[\s,]+", line.strip())
        smi = canonical(fields[0]) if fields and fields[0] else None
        if smi:
            stock.add(smi)
    return stock


def element_counts(smiles):
    """Count heavy atoms by element."""
    return Counter(atom.GetSymbol() for atom in Chem.MolFromSmiles(smiles).GetAtoms())


@dataclass
class Disconnection:
    reactants: tuple
    prob: float


@dataclass
class TreeNode:
    smiles: str
    step_prob: float = 1.0  # confidence of the disconnection that produced `children`
    children: list = field(default_factory=list)

    def steps(self):
        """Yield every internal node (a reaction step), root first."""
        if self.children:
            yield self
            for child in self.children:
                yield from child.steps()

    def leaves(self):
        """Yield every leaf node."""
        if not self.children:
            yield self
        for child in self.children:
            yield from child.leaves()


def route_score(root):
    """Product of the confidences of all steps in the route."""
    return math.prod(step.step_prob for step in root.steps())
