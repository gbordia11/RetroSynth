import dspy
import pytest

from config import Config, ValidatorConfig, load_config
from molecules import canonical
from orchestrator import PlanSynthesis, RetroOrchestrator, parse_reaction, route_from_steps

PE_RU = "[*]CC[*]"
# E(RU) = E(ethane) - E(H2) = -78.00 Hartree, below E(ethylene), so dE < 0 for ethylene -> PE.
FAKE = {canonical("CC"): -79.17, canonical("[H][H]"): -1.17, canonical("C=C"): -77.97,
        canonical("CCO"): -155.0, canonical("O"): -76.0}


def orchestrator(stock=("C=C",)):
    """Orchestrator with fake energies (dE criterion), so no ORCA, GFN2 or LLM call is needed."""
    config = Config(validator=ValidatorConfig(backend="orca", criterion="de", orca_path="no-such-orca-binary"))
    orch = RetroOrchestrator({canonical(s) for s in stock}, config)
    orch.validator.energies.energy = lambda smi: FAKE[canonical(smi)]
    return orch


# --- reading Claude's answer ------------------------------------------------------

def test_parse_reaction_reads_reactants_and_product():
    assert parse_reaction("C=C.O>>[*]CCO[*]") == ([canonical("C=C"), canonical("O")], canonical(PE_RU.replace("CC", "CCO")))


@pytest.mark.parametrize("text", ["C=C", "C=C>>not_a_smiles", ">>CC"])
def test_parse_reaction_rejects_unreadable_steps(text):
    with pytest.raises(ValueError):
        parse_reaction(text)


def test_route_from_steps_links_intermediates():
    root, issues = route_from_steps(["C=C.O>>CCO", "CCO>>C=C"], "C=C")
    assert issues == []
    assert root.smiles == canonical("C=C")
    assert [c.smiles for c in root.children] == [canonical("CCO")]
    assert sorted(c.smiles for c in root.children[0].children) == sorted([canonical("C=C"), canonical("O")])


def test_route_from_steps_reports_missing_target_and_unused_intermediates():
    root, issues = route_from_steps(["C=C.O>>CCO"], PE_RU)
    assert root is None
    assert any("no step makes the target" in i for i in issues)
    assert any("not used" in i for i in issues)


def test_signature_lists_known_byproducts():
    assert "H2O" in PlanSynthesis.instructions and "{byproducts}" not in PlanSynthesis.instructions


# --- re-check without the LLM -------------------------------------------------------

def test_check_accepts_a_favourable_polymer_route():
    result = orchestrator().check(PE_RU, ["C=C>>[*]CC[*]"])
    assert result.valid, result.issues
    assert result.steps[0].deltas["dE"] < 0


def test_check_flags_starting_materials_not_in_stock():
    result = orchestrator(stock=()).check(PE_RU, ["C=C>>[*]CC[*]"])
    assert not result.valid
    assert result.issues == [f"not in stock: {canonical('C=C')}"]


def test_check_flags_gate_failure_and_skips_its_energy():
    result = orchestrator().check("[*]CC(N)[*]", ["C=C>>[*]CC(N)[*]"])
    assert not result.valid
    assert "gate failed" in result.issues[0]
    assert result.steps[0].deltas == {}


def test_check_flags_unfavourable_step():
    orch = orchestrator(stock=("CC",))
    result = orch.check("C=C", ["CC>>C=C"])  # dehydrogenation, dE > 0 with these energies
    assert not result.valid
    assert "not favourable" in result.issues[0]


# --- tools -------------------------------------------------------------------------

def test_validate_step_tool_passes_and_fails_with_reasons():
    orch = orchestrator()
    assert orch.validate_step(PE_RU, "C=C").startswith("PASS")
    assert orch.validate_step("CC", "C=C").startswith("FAIL")  # needs H2, which is not supplied
    assert "not valid SMILES" in orch.validate_step("nope", "C=C")
    assert orch.validate_step(PE_RU, "bad smiles").startswith("FAIL")


def test_in_stock_tool():
    text = orchestrator().in_stock("C=C, CCO, nope")
    assert text.splitlines() == ["C=C: in stock", "CCO: NOT in stock", "nope: invalid SMILES"]


# --- Claude's loop (scripted LLM) -----------------------------------------------------

def scripted(steps):
    return dspy.utils.DummyLM([
        {"next_thought": "Ethylene polymerises to PE; validate the step.", "next_tool_name": "validate_step",
         "next_tool_args": {"product": PE_RU, "reactants": "C=C"}},
        {"next_thought": "Check the monomer is purchasable.", "next_tool_name": "in_stock",
         "next_tool_args": {"smiles": "C=C"}},
        {"next_thought": "The step passed and ethylene is in stock.", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "One addition step.", "steps": steps, "rationale": "Ethylene addition polymerisation, dE < 0."},
    ])


def test_orchestrator_runs_tools_and_returns_a_valid_route():
    orch = orchestrator()
    orch.lm = scripted(["C=C>>[*]CC[*]"])
    result = orch(PE_RU)
    assert result.valid, result.issues
    assert result.trajectory["tool_name_0"] == "validate_step"
    assert result.trajectory["observation_0"].startswith("PASS")
    assert "polymerisation" in result.rationale


def test_orchestrator_cannot_make_a_failing_route_valid():
    """Claude returns a step that fails the gate; the re-check rejects the route anyway."""
    orch = orchestrator()
    orch.lm = scripted(["C=C>>[*]CC(N)[*]"])
    result = orch("[*]CC(N)[*]")
    assert not result.valid
    assert any("gate failed" in issue for issue in result.issues)


def test_shipped_config_has_orchestrator_section():
    config = load_config()
    assert config.orchestrator.model.startswith("anthropic/")
    assert config.orchestrator.max_iters > 0


# --- live progress (CLI) -------------------------------------------------------------

def run_with_progress(steps):
    events = []
    orch = orchestrator()
    orch.lm = scripted(steps)
    orch.progress = lambda event, data: events.append((event, data))
    return orch(PE_RU), events


def test_progress_reports_each_iteration_in_order():
    _, events = run_with_progress(["C=C>>[*]CC[*]"])
    names = [e for e, _ in events]
    assert names[:5] == ["thinking", "thought", "tool_start", "tool_end", "thinking"]
    assert names[-3:] == ["recheck_start", "recheck_step", "recheck_done"]
    assert "answering" in names
    thought = next(d for e, d in events if e == "thought")
    assert thought["iteration"] == 1 and thought["tool"] == "validate_step"
    assert thought["args"] == {"product": PE_RU, "reactants": "C=C"}


def test_progress_tool_end_carries_the_result_and_time():
    _, events = run_with_progress(["C=C>>[*]CC[*]"])
    ends = [d for e, d in events if e == "tool_end" and d["tool"] != "finish"]
    assert [d["tool"] for d in ends] == ["validate_step", "in_stock"]
    assert ends[0]["result"].startswith("PASS")
    assert all(d["seconds"] >= 0 for d in ends)


def test_progress_recheck_reports_the_verdict():
    _, events = run_with_progress(["C=C>>[*]CC(N)[*]"])
    done = next(d for e, d in events if e == "recheck_done")
    assert done["valid"] is False and done["issues"]


def test_no_progress_callback_without_a_reporter():
    orch = orchestrator()
    orch.lm = scripted(["C=C>>[*]CC[*]"])
    assert orch(PE_RU).valid  # runs with progress=None and reports nothing


def test_printer_shows_thoughts_tool_calls_results_and_recheck():
    import io

    from orchestrator import ProgressPrinter

    out = io.StringIO()
    printer = ProgressPrinter(stream=out, color=False)
    orch = orchestrator()
    orch.lm = scripted(["C=C>>[*]CC[*]"])
    orch.progress = printer
    printer.header(PE_RU, orch.config)
    orch(PE_RU)
    text = out.getvalue()
    for expected in ["Target  [*]CC[*]", "[1] Claude is thinking", "Thought: Ethylene polymerises",
                     "Next: validate_step(product=[*]CC[*], reactants=C=C)", "-> validate_step",
                     "<- PASS", "-> in_stock", "<- C=C: in stock", "Claude is writing its answer",
                     "Re-checking Claude's 1 step(s)", "step 1: C=C -> *CC*  gate ok", "Route is VALID"]:
        assert expected in text, expected
    assert "\033[" not in text  # no colour codes when color=False
