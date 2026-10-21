"""Agent 1: template-free single-step retrosynthesis predictor (P_ss)."""

import dspy

from config import PredictorConfig
from molecules import Disconnection, canonical


class PredictPrecursors(dspy.Signature):
    """Propose single-step retrosynthetic disconnections of the target molecule.
    Reason from the molecular structure; do not rely on named reaction templates.
    Return up to k alternative disconnections, each as a set of precursor SMILES,
    with a confidence in [0, 1], ordered from most to least likely."""

    target_smiles: str = dspy.InputField(desc="SMILES of the molecule to disconnect")
    context: str = dspy.InputField(desc="extra user instructions, may be empty")
    k: int = dspy.InputField(desc="maximum number of disconnections to return")
    precursor_sets: list[list[str]] = dspy.OutputField(
        desc="each entry is a list of precursor SMILES for one disconnection"
    )
    confidences: list[float] = dspy.OutputField(
        desc="confidence in [0, 1] for each entry of precursor_sets"
    )


class SingleStepPredictor(dspy.Module):
    """Agent 1: template-free single-step predictor, returns top-K disconnections."""

    def __init__(self, config=None, backend=None):
        super().__init__()
        self.config = config or PredictorConfig()
        self.backend = backend  # optional callable: smiles -> [(reactant_smiles_list, prob), ...]
        self.lm = dspy.LM(self.config.model, max_tokens=self.config.max_tokens)
        self.llm = dspy.Predict(PredictPrecursors)

    def forward(self, smiles, context=""):
        if self.backend is not None:
            raw = self.backend(smiles)
        else:
            with dspy.context(lm=self.lm):  # this agent's model, independent of the others
                pred = self.llm(target_smiles=smiles, context=context, k=self.config.k)
            raw = zip(pred.precursor_sets, pred.confidences)
        return self._clean(smiles, raw)

    def _clean(self, target, raw):
        """Drop unparsable, trivial and duplicate disconnections; keep the top-K by confidence."""
        target_c = canonical(target)
        seen, out = set(), []
        for reactants, prob in raw:
            cans = [canonical(r) for r in reactants or []]
            if not cans or None in cans or target_c in cans:
                continue
            cans = tuple(sorted(cans))
            if cans in seen:
                continue
            seen.add(cans)
            out.append(Disconnection(cans, min(max(float(prob), 0.0), 1.0)))
        out.sort(key=lambda d: d.prob, reverse=True)
        return out[: self.config.k]
