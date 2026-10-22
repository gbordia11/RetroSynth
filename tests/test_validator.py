import os
import shutil
from collections import Counter

import dspy
import pytest

from config import ValidatorConfig
from gfn2 import Gfn2Runner
from molecules import TreeNode, canonical
from validator import (
    HARTREE_TO_KCAL,
    OrcaError,
    OrcaRunner,
    RouteValidator,
    StepCheck,
    capped,
    check_repeat_unit,
    check_step_gate,
    decompose,
    parse_energy,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC_ACID = "OC(=O)c1ccccc1O"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"
PE_RU = "[*]CC[*]"  # polyethylene repeat unit

ORCA_PATH = os.environ.get("ORCA_PATH", "orca")
requires_orca = pytest.mark.skipif(
    shutil.which(ORCA_PATH) is None,
    reason=f"ORCA not found at {ORCA_PATH!r}; install it or set ORCA_PATH",
)


def validator_without_orca(tmp_path):
    return RouteValidator(ValidatorConfig(backend="orca", criterion="de", orca_path="no-such-orca-binary", workdir=str(tmp_path)))


def aspirin_route():
    """Target aspirin, made from acetic anhydride and salicylic acid in one step."""
    return TreeNode(canonical(ASPIRIN), 0.8, [
        TreeNode(canonical(ACETIC_ANHYDRIDE)),
        TreeNode(canonical(SALICYLIC_ACID)),
    ])


# --- chemistry gate (one step, no LLM) ----------------------------------------

def test_gate_accepts_aspirin_step_with_acetic_acid_byproduct():
    reactants = {canonical(ACETIC_ANHYDRIDE): 1, canonical(SALICYLIC_ACID): 1}
    ok, byproduct, _ = check_step_gate(canonical(ASPIRIN), reactants)
    assert ok
    assert decompose(byproduct) == {"CH3COOH": 1}


def test_gate_rejects_step_when_product_needs_missing_atoms():
    ok, _, message = check_step_gate("[*]CC(N)[*]", {"C=C": 1})
    assert not ok
    assert "N" in message


def test_repeat_unit_needs_two_end_points():
    assert check_repeat_unit(PE_RU)[0]
    assert not check_repeat_unit("CCO")[0]


def test_gate_recognises_canonical_repeat_unit():
    # canonical() writes [*] as a bare *. The gate must still treat it as a repeat unit.
    ru = canonical(PE_RU)
    assert "[*]" not in ru
    ok, byproduct, _ = check_step_gate(ru, {"C=C": 1})
    assert ok and not +byproduct
    assert not check_step_gate(canonical("[*]CC"), {"C=C": 1})[0]  # one end point is not a repeat unit


def test_capped_replaces_end_points_with_hydrogen():
    assert capped(PE_RU) == canonical("CC")
    assert capped(canonical(PE_RU)) == canonical("CC")
    assert capped("[*]C(=O)CCC(=O)[*]") == canonical("O=CCCC=O")


def test_decompose_splits_backtracking_and_rejects_unknown():
    assert decompose(Counter(H=4, O=2)) == {"H2O": 2}
    assert decompose(Counter(O=1)) is None


# --- ORCA input and output (no ORCA run) -----------------------------------------

def test_orca_input_has_keyword_charge_and_atom_lines(tmp_path):
    runner = OrcaRunner(ValidatorConfig(method="B3LYP def2-SVP Opt", workdir=str(tmp_path)))
    lines = runner.input_text("CC").splitlines()
    assert lines[0] == "! B3LYP def2-SVP Opt"
    assert lines[1] == "* xyz 0 1"
    assert len(lines) - 3 == 8  # ethane: 2 C + 6 H


def test_parse_energy_reads_final_energy_and_rejects_failed_run():
    ok_output = "FINAL SINGLE POINT ENERGY      -79.5\nORCA TERMINATED NORMALLY\n"
    assert parse_energy(ok_output) == pytest.approx(-79.5)
    with pytest.raises(OrcaError):
        parse_energy("FINAL SINGLE POINT ENERGY -1.0\n")


def test_missing_orca_raises_for_energy(tmp_path):
    validator = validator_without_orca(tmp_path)
    assert not validator.energies.available()
    with pytest.raises(OrcaError, match="not found"):
        validator.step_delta_e(canonical(ASPIRIN), {canonical(ACETIC_ANHYDRIDE): 1, canonical(SALICYLIC_ACID): 1})


# --- step energy arithmetic (fake ORCA energies) --------------------------------

def test_ethylene_to_polyethylene_step_is_thermoneutral(tmp_path):
    validator = validator_without_orca(tmp_path)
    # E(RU) = E(capped ethane) - E(H2) = -79.17 + 1.17 = -78.0, which equals E(ethylene).
    fake = {canonical("CC"): -79.17, canonical("[H][H]"): -1.17, canonical("C=C"): -78.0}
    validator.energies.energy = lambda smi: fake[canonical(smi)]
    assert validator.step_delta_e(PE_RU, {"C=C": 1}) == pytest.approx(0.0)


def test_canonical_repeat_unit_step_energy_matches_bracket_form(tmp_path):
    validator = validator_without_orca(tmp_path)
    fake = {canonical("CC"): -79.17, canonical("[H][H]"): -1.17, canonical("C=C"): -78.5}
    validator.energies.energy = lambda smi: fake[canonical(smi)]
    expected = validator.step_delta_e(PE_RU, {"C=C": 1})
    assert validator.step_delta_e(canonical(PE_RU), {"C=C": 1}) == pytest.approx(expected)


def test_step_energy_includes_byproduct(tmp_path):
    validator = validator_without_orca(tmp_path)
    # C2H4 + H2O -> [*]CCO[*] + H2. dE = E(RU) + E(H2) - E(C2H4) - E(H2O) = -154 - 1 + 154 = -1 Hartree
    fake = {canonical("[H][H]"): -1.0, canonical("O"): -76.0, canonical("C=C"): -78.0, canonical("CCO"): -155.0}
    validator.energies.energy = lambda smi: fake[canonical(smi)]
    assert validator.step_delta_e("[*]CCO[*]", {"C=C": 1, "O": 1}) == pytest.approx(-HARTREE_TO_KCAL)


# --- route interface (agent behaviour) --------------------------------------------

def test_route_failing_gate_is_rejected_without_calling_llm(tmp_path):
    validator = validator_without_orca(tmp_path)
    validator.lm = dspy.utils.DummyLM([])  # any LLM call would raise, so this proves none happens
    bad = TreeNode("[*]CC(N)[*]", 0.9, [TreeNode(canonical("C=C"))])
    result = validator(bad)
    assert not result.valid
    assert "N" in result.rationale
    assert result.steps[0].gate_ok is False


def test_route_review_calls_tool_and_returns_verdict(tmp_path, monkeypatch):
    """Scripted LLM: inspects the step with the gate tool, then finishes with a verdict."""
    validator = validator_without_orca(tmp_path)
    monkeypatch.setattr(validator, "step_deltas", lambda product, reactants: {"dE": -10.0})  # favourable, no energy run
    validator.lm = dspy.utils.DummyLM([
        {
            "next_thought": "Check the acetylation step with the gate.",
            "next_tool_name": "check_step",
            "next_tool_args": {"product": canonical(ASPIRIN), "reactants": f"{ACETIC_ANHYDRIDE}, {SALICYLIC_ACID}"},
        },
        {"next_thought": "The gate passes, so I can give a verdict.", "next_tool_name": "finish", "next_tool_args": {}},
        {
            "reasoning": "One step, gate ok, acetic acid by-product.",
            "valid": True,
            "rationale": "Acetylation of salicylic acid with acetic anhydride is balanced.",
        },
    ])
    result = validator(aspirin_route())
    assert result.valid is True
    assert result.steps[0].gate_ok is True
    assert "balanced" in result.rationale


# --- ORCA-backed validation (runs only when the ORCA binary is installed) -------------

@requires_orca
def test_ethylene_polymerisation_step_energy_from_orca(tmp_path):
    """Real ORCA runs on ethane, H2 and ethylene. Polymerisation of ethylene is exothermic."""
    validator = RouteValidator(ValidatorConfig(backend="orca", criterion="de", orca_path=ORCA_PATH, workdir=str(tmp_path)))
    delta = validator.step_delta_e(PE_RU, {"C=C": 1})
    assert -40 < delta < 0  # kcal/mol per repeat unit; literature is roughly -20 to -25
    assert len(list(tmp_path.glob("*.out"))) == 3  # one output per species, ethane, H2, ethylene


# --- thermodynamic verdict with GFN2-xTB (default backend) -------------------------

requires_tblite = pytest.mark.skipif(not Gfn2Runner.available(), reason="tblite is not installed")


def hydrogenation_route():
    """Ethylene + H2 -> ethane, one step. Exothermic, so it should pass the energy check."""
    return TreeNode("CC", 0.9, [TreeNode("C=C"), TreeNode("[H][H]")])


def dehydrogenation_route():
    """Ethane -> ethylene + H2, one step. Endothermic, so the energy check must reject it."""
    return TreeNode("C=C", 0.9, [TreeNode("CC")])


@requires_tblite
def test_exothermic_route_is_approved_when_llm_agrees(tmp_path):
    validator = RouteValidator(ValidatorConfig(workdir=str(tmp_path)))
    validator.lm = dspy.utils.DummyLM([
        {"next_thought": "Energies favour the step.", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "Hydrogenation is exothermic.", "valid": True, "rationale": "Exothermic hydrogenation."},
    ])
    result = validator(hydrogenation_route())
    assert result.favourable is True
    assert result.valid is True
    assert result.steps[0].deltas["dG"] < 0
    assert result.steps[0].deltas["dH"] < 0


@requires_tblite
def test_llm_can_reject_a_route_that_passed_the_energy_check(tmp_path):
    validator = RouteValidator(ValidatorConfig(workdir=str(tmp_path)))
    validator.lm = dspy.utils.DummyLM([
        {"next_thought": "Looks fine.", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "The reagent is impractical.", "valid": False, "rationale": "Hydrogen gas is impractical here."},
    ])
    result = validator(hydrogenation_route())
    assert result.favourable is True
    assert result.valid is False


@requires_tblite
def test_endothermic_route_is_rejected_without_calling_llm(tmp_path):
    validator = RouteValidator(ValidatorConfig(workdir=str(tmp_path)))
    validator.lm = dspy.utils.DummyLM([])  # any LLM call would raise, so this proves none happens
    result = validator(dehydrogenation_route())
    assert result.valid is False
    assert result.favourable is False
    assert "not thermodynamically favourable" in result.rationale
    assert result.steps[0].deltas["dG"] > 0


# --- criteria and ceiling temperature (no energy runs) ------------------------------

def test_default_criteria_are_dg_and_dh(tmp_path):
    validator = RouteValidator(ValidatorConfig(workdir=str(tmp_path)))
    assert validator.criteria == ["dg", "dh"]


def test_step_must_pass_every_criterion(tmp_path, monkeypatch):
    validator = RouteValidator(ValidatorConfig(workdir=str(tmp_path)))
    validator.lm = dspy.utils.DummyLM([])  # rejected before the LLM review
    # dG < 0 but dH > 0: an entropy-driven step, which the dH criterion rejects.
    monkeypatch.setattr(validator, "step_deltas", lambda product, reactants: {"dG": -5.0, "dH": 2.0})
    result = validator(aspirin_route())
    assert not result.valid and not result.favourable
    assert "dG/dH" in result.rationale and "dH=+2.0" in result.rationale


def test_ceiling_temperature_from_dg_and_dh():
    # dS = (dH - dG) / T = (-20 + 10) / 298.15 kcal/(mol K), so Tc = dH / dS = 2 * 298.15 K.
    step = StepCheck("p", ["r"], True, "", deltas={"dG": -10.0, "dH": -20.0})
    assert step.ceiling_temperature == pytest.approx(2 * 298.15)


def test_no_ceiling_temperature_without_both_values_or_for_positive_entropy():
    assert StepCheck("p", ["r"], True, "", deltas={"dG": -10.0}).ceiling_temperature is None
    assert StepCheck("p", ["r"], True, "", deltas={"dG": -10.0, "dH": -5.0}).ceiling_temperature is None  # dS > 0
