from collections import Counter

import dspy
import pytest

from config import ValidatorConfig
from molecules import canonical
from validator import (
    HARTREE_TO_KCAL,
    OrcaError,
    OrcaRunner,
    RouteValidator,
    balance,
    check_repeat_unit,
    decompose,
    format_species,
    formula,
    formula_str,
    parse_energy,
    parse_species,
)

PE_RU = "[*]CC[*]"  # polyethylene
NYLON_66_RU = "[*]NCCCCCCNC(=O)CCCCC(=O)[*]"
ADIPIC_ACID = "OC(=O)CCCCC(=O)O"
HEXAMETHYLENEDIAMINE = "NCCCCCCN"


def no_orca_validator(tmp_path):
    config = ValidatorConfig(orca_path="no-such-orca-binary", workdir=str(tmp_path))
    return RouteValidator(config)


# --- species parsing and formulae ----------------------------------------

def test_parse_species_reads_counts_and_canonicalises():
    species = parse_species("C=C, OC(=O)C:2, C=C")
    assert species == {canonical("C=C"): 2, canonical("OC(=O)C"): 2}


def test_parse_species_rejects_invalid_and_star_input():
    with pytest.raises(ValueError):
        parse_species("not_a_smiles")
    with pytest.raises(ValueError):
        parse_species("[*]CC[*]")


def test_format_species_round_trips():
    species = parse_species("C=C, OC(=O)C:2")
    assert parse_species(format_species(species)) == species


def test_formula_ignores_end_points():
    assert formula(PE_RU) == formula("C=C")
    assert formula_str(formula("OC(=O)CCCCC(=O)O")) == "C6H10O4"


def test_formula_str_uses_hill_order_and_marks_empty():
    assert formula_str(formula("CCO")) == "C2H6O"
    assert formula_str(formula("O")) == "H2O"
    assert formula_str(Counter()) == "(none)"


# --- repeat units and mass balance -----------------------------------------

def test_check_repeat_unit_requires_two_end_points():
    assert check_repeat_unit(PE_RU)[0]
    assert not check_repeat_unit("CCO")[0]
    assert not check_repeat_unit("[*]CC")[0]
    assert not check_repeat_unit("not_a_smiles")[0]


def test_addition_polymerisation_is_balanced_with_no_byproduct():
    ok, byproduct, _ = balance(parse_species("C=C"), PE_RU)
    assert ok
    assert not +byproduct


def test_condensation_polymer_has_two_water_byproducts():
    monomers = parse_species(f"{ADIPIC_ACID}, {HEXAMETHYLENEDIAMINE}")
    ok, byproduct, _ = balance(monomers, NYLON_66_RU)
    assert ok
    assert formula_str(byproduct) == "H4O2"
    assert decompose(byproduct) == {"H2O": 2}


def test_balance_fails_when_repeat_unit_needs_missing_atoms():
    ok, _, message = balance(parse_species("C=C"), "[*]CC(N)[*]")
    assert not ok
    assert "N" in message


def test_decompose_rejects_unknown_byproduct():
    assert decompose(formula("CCC")) is None  # C3H8 is not a known by-product
    assert decompose(Counter(O=1)) is None  # a lone oxygen atom is not a molecule
    assert decompose(Counter(H=4, O=2)) == {"H2O": 2}  # backtracking: greedy H2 first fails


# --- ORCA input and output -------------------------------------------------

def test_orca_input_has_keyword_charge_and_one_line_per_atom(tmp_path):
    runner = OrcaRunner(ValidatorConfig(method="B3LYP def2-SVP Opt", workdir=str(tmp_path)))
    lines = runner.input_text("CC").splitlines()
    assert lines[0] == "! B3LYP def2-SVP Opt"
    assert lines[1] == "* xyz 0 1"
    assert lines[-1] == "*"
    assert len(lines) - 3 == 8  # ethane: 2 C + 6 H


def test_orca_input_sets_multiplicity_for_radicals(tmp_path):
    runner = OrcaRunner(ValidatorConfig(workdir=str(tmp_path)))
    assert runner.input_text("[CH3]").splitlines()[1] == "* xyz 0 2"


def test_parse_energy_reads_final_energy():
    text = "...\nFINAL SINGLE POINT ENERGY      -79.123456789\n...\nORCA TERMINATED NORMALLY\n"
    assert parse_energy(text) == pytest.approx(-79.123456789)


def test_parse_energy_rejects_failed_run():
    with pytest.raises(OrcaError):
        parse_energy("FINAL SINGLE POINT ENERGY -1.0\n")
    with pytest.raises(OrcaError):
        parse_energy("ORCA TERMINATED NORMALLY\n")


def test_missing_orca_is_reported_not_raised(tmp_path):
    validator = no_orca_validator(tmp_path)
    assert not validator.orca.available()
    assert "not found" in validator.orca_energy("CC")
    assert "not found" in validator.reaction_energy("C=C", PE_RU)


# --- reaction energy arithmetic (fake ORCA energies) -----------------------

def test_reaction_energy_uses_capped_repeat_unit_minus_h2(tmp_path):
    validator = no_orca_validator(tmp_path)
    # Ethylene -> polyethylene repeat unit: E(RU) = E(capped) - E(H2) = -78.0, so dE = 0.
    fake = {canonical("CC"): -79.17, canonical("[H][H]"): -1.17, canonical("C=C"): -78.0}
    validator.orca.energy = lambda smi: fake[canonical(smi)]
    assert validator.reaction_energy_kcal("C=C", PE_RU) == pytest.approx(0.0)


def test_reaction_energy_includes_byproduct_energy(tmp_path):
    validator = no_orca_validator(tmp_path)
    # Toy route: C2H4 + H2O -> C2H4O (repeat unit "[*]CCO[*]") + H2, so the by-product is H2.
    # E(RU) = E(capped ethanol) - E(H2) = -155 + 1 = -154
    # dE = E(RU) + E(H2) - E(C2H4) - E(H2O) = -154 - 1 + 154 = -1 Hartree
    fake = {
        canonical("[H][H]"): -1.0,
        canonical("O"): -76.0,
        canonical("C=C"): -78.0,
        canonical("CCO"): -155.0,
    }
    validator.orca.energy = lambda smi: fake[canonical(smi)]
    delta = validator.reaction_energy_kcal("C=C, O", "[*]CCO[*]")
    assert delta == pytest.approx(-HARTREE_TO_KCAL)


# --- agent behaviour ---------------------------------------------------------

def test_gate_rejects_without_calling_the_llm(tmp_path):
    validator = no_orca_validator(tmp_path)
    result = validator.forward("C=C", "[*]CC(N)[*]")  # needs nitrogen, none supplied
    assert not result.valid
    assert "N" in result.rationale


def test_gate_rejects_invalid_monomer_text(tmp_path):
    validator = no_orca_validator(tmp_path)
    result = validator.forward("not_a_smiles", PE_RU)
    assert not result.valid


def test_react_agent_calls_tools_and_returns_verdict(tmp_path):
    """Scripted LLM: calls the mass-balance tool, then finishes. Checks the tool loop end to end."""
    validator = no_orca_validator(tmp_path)
    responses = [
        {
            "next_thought": "Check the mass balance first.",
            "next_tool_name": "check_mass_balance",
            "next_tool_args": {"monomers": "C=C", "polymer": PE_RU},
        },
        {
            "next_thought": "Balanced with no by-product, so the route is sound.",
            "next_tool_name": "finish",
            "next_tool_args": {},
        },
        {
            "reasoning": "The mass-balance tool reports exact balance with no by-product.",
            "valid": True,
            "rationale": "Mass balance is exact and no by-product is formed.",
        },
    ]
    validator.lm = dspy.utils.DummyLM(responses)
    result = validator.forward("C=C", PE_RU)
    assert result.valid is True
    assert "exact" in result.rationale
