import pytest

from config import ValidatorConfig
from gfn2 import BOHR_PER_ANGSTROM, Gfn2Runner
from molecules import canonical
from validator import HARTREE_TO_KCAL, RouteValidator

requires_tblite = pytest.mark.skipif(not Gfn2Runner.available(), reason="tblite is not installed")


@pytest.fixture
def validator(tmp_path):
    return RouteValidator(ValidatorConfig(backend="gfn2", workdir=str(tmp_path)))


@requires_tblite
def test_relaxation_lowers_the_energy_below_the_starting_geometry():
    runner = Gfn2Runner()
    numbers, positions, charge, unpaired = runner.geometry("CC")
    start = runner.singlepoint(numbers, positions, charge, unpaired).get("energy")
    relaxed = runner.energy("CC")
    assert relaxed <= start + 1e-8


@requires_tblite
def test_energy_is_cached_per_molecule():
    runner = Gfn2Runner()
    first = runner.energy("C=C")
    assert runner.energy("C=C") == first
    assert runner.energy(canonical("C=C")) == first


@requires_tblite
def test_hydrogenation_of_ethylene_is_exothermic(validator):
    # C2H4 + H2 -> C2H6. Known to be exothermic, so dE must be below zero.
    delta = validator.step_delta_e("CC", {"C=C": 1, "[H][H]": 1})
    assert delta < 0


@requires_tblite
def test_dehydrogenation_of_ethane_is_endothermic(validator):
    # C2H6 -> C2H4 + H2 is the reverse of hydrogenation, so dE must be above zero.
    delta = validator.step_delta_e("C=C", {"CC": 1})
    assert delta > 0


@requires_tblite
def test_step_energies_are_opposite_for_forward_and_reverse(validator):
    forward = validator.step_delta_e("CC", {"C=C": 1, "[H][H]": 1})
    reverse = validator.step_delta_e("C=C", {"CC": 1})
    assert forward == pytest.approx(-reverse, abs=1e-6)
    assert abs(forward) * HARTREE_TO_KCAL > 1  # not a numerical zero


@requires_tblite
def test_hydrogenation_free_energy_is_negative(validator):
    assert validator.step_delta_g("CC", {"C=C": 1, "[H][H]": 1}) < 0


@requires_tblite
def test_dehydrogenation_free_energy_is_positive(validator):
    assert validator.step_delta_g("C=C", {"CC": 1}) > 0


@requires_tblite
def test_free_energy_differs_from_electronic_energy(validator):
    # The thermal and entropy terms must change the answer by several kcal/mol for this step.
    dg = validator.step_delta_g("CC", {"C=C": 1, "[H][H]": 1})
    de = validator.step_delta_e("CC", {"C=C": 1, "[H][H]": 1})
    assert abs(dg - de) > 5


@requires_tblite
def test_hydrogenation_enthalpy_is_negative_and_reverses(validator):
    forward = validator.step_delta_h("CC", {"C=C": 1, "[H][H]": 1})
    reverse = validator.step_delta_h("C=C", {"CC": 1})
    assert forward < 0
    assert forward == pytest.approx(-reverse, abs=1e-6)


@requires_tblite
def test_free_energy_and_enthalpy_share_one_hessian(validator, monkeypatch):
    validator.energies.free_energy("C=C")
    monkeypatch.setattr(validator.energies, "hessian", lambda *a: pytest.fail("Hessian recomputed"))
    validator.energies.enthalpy("C=C")


@requires_tblite
def test_polyethylene_from_ethylene_is_exergonic_and_exothermic(validator):
    """The canonical repeat unit (bare * end points) goes through the full dG/dH path."""
    deltas = validator.step_deltas(canonical("[*]CC[*]"), {"C=C": 1})
    assert deltas["dH"] < 0  # literature about -22 kcal/mol per repeat unit
    assert deltas["dG"] < 0
    assert deltas["dG"] > deltas["dH"]  # the entropy loss on polymerisation opposes it


@requires_tblite
def test_free_energy_is_cached_per_molecule():
    runner = Gfn2Runner()
    first = runner.free_energy("[H][H]")
    assert runner.free_energy("[H][H]") == first


def test_dg_criterion_needs_gfn2(tmp_path):
    with pytest.raises(ValueError, match="needs backend 'gfn2'"):
        RouteValidator(ValidatorConfig(backend="orca", criterion="dg", workdir=str(tmp_path)))


def test_unknown_criterion_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown criterion"):
        RouteValidator(ValidatorConfig(criterion="dg, ds", workdir=str(tmp_path)))


def test_dh_criterion_needs_gfn2(tmp_path):
    with pytest.raises(ValueError, match="needs backend 'gfn2'"):
        RouteValidator(ValidatorConfig(backend="orca", criterion="dh", workdir=str(tmp_path)))


def test_criteria_list_is_parsed(tmp_path):
    assert RouteValidator(ValidatorConfig(criterion=" dg ,dh, de ", workdir=str(tmp_path))).criteria == ["dg", "dh", "de"]
    assert RouteValidator(ValidatorConfig(backend="orca", criterion="de", workdir=str(tmp_path))).criteria == ["de"]


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown validator backend"):
        RouteValidator(ValidatorConfig(backend="xtb-typo", workdir=str(tmp_path)))


def test_bohr_conversion_constant():
    assert BOHR_PER_ANGSTROM == pytest.approx(1.8897261, rel=1e-6)


def test_missing_tblite_is_reported_by_available(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_tblite(name, *args, **kwargs):
        if name.startswith("tblite"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tblite)
    assert not Gfn2Runner.available()
