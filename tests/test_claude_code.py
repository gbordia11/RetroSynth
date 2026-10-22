"""The claude_code backend: the orchestrator reaches Claude through a fake `claude` CLI."""

import json
import stat
import sys

import dspy
import pytest

from config import Config, OrchestratorConfig, ValidatorConfig, load_config
from molecules import canonical
from dspy.lm15 import ConfigurationError, Message, ProviderError, Request

from orchestrator import ClaudeCodeEngine, RetroOrchestrator, make_lm

PE_RU = "[*]CC[*]"
FAKE = {canonical("CC"): -79.17, canonical("[H][H]"): -1.17, canonical("C=C"): -77.97}

# A stand-in for `claude -p --output-format json`: replays scripted replies in order and records
# each call (arguments, stdin, whether ANTHROPIC_API_KEY reached it) as one JSON line.
FAKE_CLI = """#!{python}
import json, os, sys
state = {state!r}
with open(state) as fh:
    data = json.load(fh)
call = {{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "api_key": "ANTHROPIC_API_KEY" in os.environ, "cwd": os.getcwd()}}
data["calls"].append(call)
reply = data["replies"].pop(0) if data["replies"] else {{"is_error": True, "result": "no scripted reply left"}}
with open(state, "w") as fh:
    json.dump(data, fh)
if reply.get("exit"):
    sys.stderr.write(reply.get("stderr", ""))
    sys.exit(reply["exit"])
print(json.dumps({{"type": "result", "usage": {{"input_tokens": 5, "output_tokens": 2}}, **reply}}))
"""


@pytest.fixture
def fake_cli(tmp_path):
    """Returns (path to the fake claude, function that sets its replies, function that reads its calls)."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"replies": [], "calls": []}))
    cli = tmp_path / "claude"
    cli.write_text(FAKE_CLI.format(python=sys.executable, state=str(state)))
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)

    def set_replies(replies):
        data = json.loads(state.read_text())
        data["replies"] = [r if isinstance(r, dict) else {"is_error": False, "result": r} for r in replies]
        state.write_text(json.dumps(data))

    def calls():
        return json.loads(state.read_text())["calls"]

    return str(cli), set_replies, calls


def fields(**values):
    """A reply in dspy's ChatAdapter format: [[ ## name ## ]] blocks, then [[ ## completed ## ]]."""
    blocks = [f"[[ ## {k} ## ]]\n{v if isinstance(v, str) else json.dumps(v)}" for k, v in values.items()]
    return "\n\n".join(blocks + ["[[ ## completed ## ]]"])


def claude_code_lm(cli, model="anthropic/claude-sonnet-5-5"):
    # dspy caches LM answers (also on disk); these tests need every call to reach the fake CLI.
    return make_lm(OrchestratorConfig(backend="claude_code", model=model, claude_path=cli)).copy(cache=False)


def request(text="hi"):
    return Request(model="anthropic/claude-sonnet-5-5", messages=(Message.user(text),))


# --- the engine on its own ------------------------------------------------------------

def test_lm_sends_system_prompt_and_body_and_strips_the_api_key(fake_cli, monkeypatch):
    cli, set_replies, calls = fake_cli
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-passed")
    set_replies([fields(answer="Paris")])
    with dspy.context(lm=claude_code_lm(cli)):
        assert dspy.Predict("question -> answer")(question="Capital of France?").answer == "Paris"
    call = calls()[0]
    assert call["api_key"] is False
    argv = call["argv"]
    assert argv[:1] == ["-p"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"  # provider prefix removed
    assert argv[argv.index("--tools") + 1] == ""  # Claude Code's own tools are off
    assert "--strict-mcp-config" in argv and "--no-session-persistence" in argv
    system = argv[argv.index("--system-prompt") + 1]
    assert "question" in system and "answer" in system  # dspy's field instructions
    assert "Capital of France?" in call["stdin"]  # the prompt goes on stdin, not the command line
    assert "Capital of France?" not in " ".join(argv)


def test_lm_runs_outside_the_project_folder(fake_cli):
    import tempfile
    from pathlib import Path

    cli, set_replies, calls = fake_cli
    set_replies([fields(answer="ok")])
    with dspy.context(lm=claude_code_lm(cli)):
        dspy.Predict("question -> answer")(question="q")
    assert Path(calls()[0]["cwd"]).resolve() == Path(tempfile.gettempdir()).resolve()


def test_lm_raises_a_clear_error_when_the_cli_reports_one(fake_cli):
    cli, set_replies, _ = fake_cli
    set_replies([{"is_error": True, "result": "Not logged in"}])
    with pytest.raises(ProviderError, match="Not logged in"):
        ClaudeCodeEngine(cli).complete(request())


def test_lm_raises_when_the_cli_exits_without_json(fake_cli):
    cli, set_replies, _ = fake_cli
    set_replies([{"exit": 2, "stderr": "unknown option --tools"}])
    with pytest.raises(ProviderError, match="unknown option"):
        ClaudeCodeEngine(cli).complete(request())


def test_lm_reports_a_missing_cli(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        ClaudeCodeEngine(str(tmp_path / "no-such-claude")).complete(request())


def test_engine_returns_the_cli_answer_and_usage(fake_cli):
    cli, set_replies, _ = fake_cli
    set_replies(["hello"])
    response = ClaudeCodeEngine(cli).complete(request())
    assert response.message.text == "hello"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (5, 2)


def test_engine_state_round_trips():
    engine = ClaudeCodeEngine("/opt/claude", 30)
    again = ClaudeCodeEngine.load_state(engine.dump_state())
    assert (again.claude_path, again.timeout) == ("/opt/claude", 30)


def test_no_deprecated_dspy_interface_is_used(fake_cli):
    """The engine API is the one dspy 3.5 keeps; the old BaseLM.forward() path warns."""
    import warnings

    cli, set_replies, _ = fake_cli
    set_replies([fields(answer="ok")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with dspy.context(lm=claude_code_lm(cli)):
            dspy.Predict("question -> answer")(question="q")


# --- choosing the backend -----------------------------------------------------------

def test_api_backend_is_the_default():
    assert load_config().orchestrator.backend == "api"
    lm = make_lm(OrchestratorConfig())
    assert isinstance(lm, dspy.LM) and not isinstance(lm.engine, ClaudeCodeEngine)


def test_claude_code_backend_builds_an_lm_with_the_cli_engine():
    lm = make_lm(OrchestratorConfig(backend="claude_code", claude_path="/opt/claude", timeout=30))
    assert isinstance(lm, dspy.LM) and isinstance(lm.engine, ClaudeCodeEngine)
    assert (lm.engine.claude_path, lm.engine.timeout) == ("/opt/claude", 30)
    argv = ClaudeCodeEngine().command("anthropic/claude-sonnet-5-5", "")
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown orchestrator backend"):
        make_lm(OrchestratorConfig(backend="cli-typo"))


# --- the whole orchestrator through the CLI ------------------------------------------

def test_orchestrator_runs_its_react_loop_through_the_cli(fake_cli):
    cli, set_replies, calls = fake_cli
    config = Config(
        validator=ValidatorConfig(backend="orca", criterion="de", orca_path="no-such-orca-binary"),
        orchestrator=OrchestratorConfig(backend="claude_code", claude_path=cli),
    )
    orch = RetroOrchestrator({canonical("C=C")}, config)
    orch.validator.energies.energy = lambda smi: FAKE[canonical(smi)]
    orch.lm = orch.lm.copy(cache=False)
    set_replies([
        fields(next_thought="Validate ethylene to PE.", next_tool_name="validate_step",
               next_tool_args={"product": PE_RU, "reactants": "C=C"}),
        fields(next_thought="It passed.", next_tool_name="finish", next_tool_args={}),
        fields(reasoning="One step.", steps=["C=C>>[*]CC[*]"], rationale="Ethylene addition polymerisation."),
    ])
    result = orch(PE_RU)
    assert result.valid, result.issues
    assert result.trajectory["observation_0"].startswith("PASS")
    assert len(calls()) == 3  # two ReAct rounds and the final answer, one CLI call each
    assert "PASS" in calls()[1]["stdin"]  # the tool result is sent back to Claude in the next call


# --- replies that drift from dspy's format -----------------------------------------------

REACT_PROMPT = ("[[ ## target ## ]]\n[*]CC[*]\n\nRespond with the corresponding output fields, starting with the field "
                "`[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]`, then `[[ ## next_tool_args ## ]]`, "
                "and then ending with the marker for `[[ ## completed ## ]]`.")
GOOD = fields(next_thought="Check stock.", next_tool_name="in_stock", next_tool_args={"smiles": "C=C"})
# What Claude Code returned in a real nylon-6,6 run: the thought as plain text, then tool-call markup.
DRIFTED = ('Adipoyl chloride is not in stock, so check other reagents.\n\n<invoke name="in_stock">\n'
           '<parameter name="smiles">NC(N)=O</parameter>\n</invoke>')


def test_format_problem_names_missing_fields_and_tool_markup():
    from orchestrator import format_problem

    assert format_problem(REACT_PROMPT, GOOD) == ""
    problem = format_problem(REACT_PROMPT, DRIFTED)
    assert "[[ ## next_thought ## ]]" in problem and "[[ ## completed ## ]]" in problem
    assert "tool-call markup" in problem
    assert format_problem("no dspy instructions here", DRIFTED) == ""  # other prompts are not checked


def test_engine_asks_again_when_a_reply_drifts_from_the_format(fake_cli):
    cli, set_replies, calls = fake_cli
    set_replies([DRIFTED, GOOD])
    response = ClaudeCodeEngine(cli).complete(Request(model="m", messages=(Message.user(REACT_PROMPT),)))
    assert response.message.text == GOOD
    assert len(calls()) == 2
    retry = calls()[1]["stdin"]
    assert DRIFTED in retry and "did not follow the required format" in retry
    assert (response.usage.input_tokens, response.usage.output_tokens) == (10, 4)  # both calls counted


def test_engine_does_not_retry_a_good_reply(fake_cli):
    cli, set_replies, calls = fake_cli
    set_replies([GOOD])
    ClaudeCodeEngine(cli).complete(Request(model="m", messages=(Message.user(REACT_PROMPT),)))
    assert len(calls()) == 1


def test_system_prompt_says_there_are_no_native_tools(fake_cli):
    cli, set_replies, calls = fake_cli
    set_replies([GOOD])
    ClaudeCodeEngine(cli).complete(Request(model="m", messages=(Message.user(REACT_PROMPT),), system="dspy rules"))
    argv = calls()[0]["argv"]
    system = argv[argv.index("--system-prompt") + 1]
    assert system.startswith("dspy rules") and "no native tools" in system and "<invoke>" in system


def test_cli_prints_a_clear_message_when_a_reply_cannot_be_read(tmp_path, monkeypatch, capsys):
    import orchestrator
    from dspy.utils.exceptions import AdapterParseError

    from orchestrator import PlanSynthesis

    stock = tmp_path / "stock.smi"
    stock.write_text("C=C\n")

    def broken(self, target, context=""):
        raise AdapterParseError("ChatAdapter", PlanSynthesis, DRIFTED)

    monkeypatch.setattr(orchestrator.RetroOrchestrator, "forward", broken)
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", PE_RU, "--stock", str(stock), "--quiet"])
    with pytest.raises(SystemExit) as exit_info:
        orchestrator.main()
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "Stopped: Claude's reply could not be read as steps, rationale" in err
    assert '<invoke name="in_stock">' in err
    assert "Traceback" not in err


def test_cli_prints_a_clear_message_when_the_lm_call_fails(tmp_path, monkeypatch, capsys):
    """For example the safeguard refusal seen in a live nylon-6,6 run: no traceback, the provider's message."""
    import orchestrator
    from dspy.utils.exceptions import LMProviderError

    stock = tmp_path / "stock.smi"
    stock.write_text("C=C\n")
    message = "Claude Code CLI returned an error: API Error: safeguards flagged this message"

    def refused(self, target, context=""):
        raise LMProviderError(message)

    monkeypatch.setattr(orchestrator.RetroOrchestrator, "forward", refused)
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", PE_RU, "--stock", str(stock), "--quiet"])
    with pytest.raises(SystemExit) as exit_info:
        orchestrator.main()
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "Stopped: the call to Claude failed." in err and "safeguards flagged" in err
    assert "Traceback" not in err
