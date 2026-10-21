import sys
from pathlib import Path

# Make the flat project modules (agent, molecules, predictor, ...) importable from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from molecules import canonical  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC_ACID = "OC(=O)c1ccccc1O"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"
ACETIC_ACID = "CC(=O)O"


@pytest.fixture
def stock():
    return {canonical(s) for s in (SALICYLIC_ACID, ACETIC_ANHYDRIDE, ACETIC_ACID)}


@pytest.fixture
def aspirin_table():
    """Stub P_ss output for aspirin and acetic anhydride, including bad entries."""
    return {
        ASPIRIN: [
            (["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"], 0.8),
            (["OC(=O)c1ccccc1O", "CC(=O)O"], 0.6),
            (["OC(=O)c1ccccc1O", "CC(=O)OC(C)=O"], 0.5),  # duplicate
            (["not_a_smiles", "CC(=O)O"], 0.9),           # invalid SMILES
            ([ASPIRIN], 0.99),                            # trivial self-disconnection
        ],
        ACETIC_ANHYDRIDE: [(["CC(=O)O"], 0.4)],
    }
