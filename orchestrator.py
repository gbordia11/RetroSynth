"""Claude as a tool-calling orchestrator for polymer (or small-molecule) retrosynthesis.

Claude reads the target, proposes monomers and a forward reaction path, and checks each step with
tools that wrap the validator (agent 5): the chemistry gate, then the energy changes from the
[validator] backend and criterion (dG and dH from GFN2-xTB by default). It revises steps that fail
and returns the path as reaction SMILES.

Claude's answer is then re-checked without the LLM: every step must pass the gate and have every
configured energy change below zero, and every starting material must be in stock. Claude can
propose a route, but only the tools can make it valid.

While it runs, the CLI prints each step as it happens: Claude's thoughts, every tool call and its
result, and the re-check of each reaction step. Pass --quiet to print only the result.

Claude is reached in one of two ways, set by `backend` in [orchestrator]:
- api (default): the Anthropic API through dspy, billed to ANTHROPIC_API_KEY.
- claude_code: the local Claude Code CLI (`claude -p`), using its own login, for example a Claude
  subscription. No API key is needed. Pass --claude-code to choose it for one run.

Usage:
    python orchestrator.py "[*]CC[*]" --stock stock.smi [--context TEXT] [--config config.ini] [--quiet] [--claude-code]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field, replace

import dspy
from dspy.clients.engines.base import validate_request
from dspy.lm15 import ConfigurationError, Message, ProviderError, Response, TextPart, Usage, response_to_events
from dspy.lm15 import TimeoutError as LMTimeoutError
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError, LMError

from config import DEFAULT_CONFIG_PATH, Config, load_config
from molecules import TreeNode, canonical, load_stock
from ranker import RouteRanker
from renderer import RouteRenderer
from validator import (
    ENERGY_ERRORS,
    KNOWN_BYPRODUCTS,
    RouteValidator,
    StepCheck,
    check_step_gate,
    format_deltas,
    parse_species,
    route_steps,
)


class PlanSynthesis(dspy.Signature):
    """Plan a synthesis of the target from purchasable starting materials.
    The target is a molecule, or a polymer repeat unit in PSMILES with two [*] end points.
    For a polymer, the last step turns the monomers of one repeat unit into the repeat unit,
    for example C=C>>[*]CC[*] or Nc1ccc(N)cc1.O=C(Cl)c1ccc(C(=O)Cl)cc1>>[*]Nc1ccc(NC(=O)c2ccc(C([*])=O)cc2)cc1.
    Write reactants only; by-products are found by atom balance and must be small molecules
    from this list: {byproducts}.
    Check every step with validate_step before you return it; it runs the atom-balance gate and
    computes the energy changes, and a step passes only if all of them are below zero. Check
    starting materials with in_stock. If a step fails, revise the monomers or the step and validate
    again. Return only steps that passed validate_step."""

    target: str = dspy.InputField(desc="SMILES of the target molecule, or PSMILES of the polymer repeat unit")
    context: str = dspy.InputField(desc="extra instructions from the chemist, may be empty")
    steps: list[str] = dspy.OutputField(
        desc="reaction SMILES in forward order, 'reactant1.reactant2>>product'; the last product is the target"
    )
    rationale: str = dspy.OutputField(desc="why these monomers and steps, citing the energy changes")


# dspy reads the instructions when the class is built, so the by-product list is filled in here.
PlanSynthesis = PlanSynthesis.with_instructions(PlanSynthesis.__doc__.format(byproducts=", ".join(KNOWN_BYPRODUCTS)))


@dataclass
class OrchestratorResult:
    valid: bool  # every step passed the gate and the energy criteria, and every leaf is in stock
    rationale: str  # Claude's explanation
    steps: list = field(default_factory=list)  # StepCheck, in forward order (starting materials first)
    issues: list = field(default_factory=list)  # why the route is not valid
    root: TreeNode | None = None  # the route as a tree, as the search agent would return it
    trajectory: dict = field(default_factory=dict)  # Claude's thoughts and tool calls


class ClaudeCodeEngine:
    """dspy engine that answers each request through the local Claude Code CLI instead of the Anthropic API.

    It implements dspy's engine contract, complete(Request) -> Response, and is used as
    dspy.LM(model, engine=ClaudeCodeEngine()). Each call runs `claude -p` once, with all of Claude
    Code's own tools turned off, so Claude only answers the prompt; the orchestrator's tools still run
    here, in dspy's ReAct loop. dspy's system message replaces Claude Code's system prompt, and the
    conversation goes in on stdin.

    ANTHROPIC_API_KEY is removed from the CLI's environment, so the CLI uses its own login (for
    example a Claude subscription) rather than billing the API key.
    """

    def __init__(self, claude_path="claude", timeout=600):
        self.claude_path = claude_path
        self.timeout = timeout

    def command(self, model, system):
        return [
            self.claude_path, "-p",
            "--output-format", "json",
            "--model", model.split("/", 1)[-1],  # the CLI takes names without dspy's provider prefix
            "--tools", "",  # no Claude Code tools: Claude answers, the orchestrator's tools do the work
            "--strict-mcp-config",  # and no MCP servers
            "--no-session-persistence",
            "--system-prompt", system,
        ]

    def complete(self, request):
        validate_request(request)
        system = request.system if isinstance(request.system, str) else "".join(p.text for p in request.system or ())
        conversation = [(getattr(m.role, "value", m.role), m.text or "") for m in request.messages]
        if len(conversation) == 1:
            body = conversation[0][1]
        else:  # few-shot demos or history: keep the turns labelled
            body = "\n\n".join(f"{role}:\n{text}" for role, text in conversation)
        system += NO_TOOL_MARKUP
        text, tokens_in, tokens_out = self._ask(request.model, system, body)
        problem = format_problem(body, text)
        if problem:
            # Claude drifted from dspy's reply format (seen in long ReAct runs). Show it its reply
            # and what was wrong, and ask once more; if that also fails, dspy reports the parse error.
            retry = (f"{body}\n\nassistant:\n{text}\n\nuser:\nYour reply above did not follow the required "
                     f"format: {problem}. Reply again from the start, with every required [[ ## field ## ]] section "
                     f"in order and no tool-call markup.")
            text, more_in, more_out = self._ask(request.model, system, retry)
            tokens_in, tokens_out = tokens_in + more_in, tokens_out + more_out
        return Response(
            id=None,
            model=request.model,
            message=Message.assistant([TextPart(text)]),
            finish_reason="stop",
            usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out),
        )

    def _ask(self, model, system, body):
        """One `claude -p` call. Returns (reply text, input tokens, output tokens)."""
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        try:
            proc = subprocess.run(
                self.command(model, system), input=body, capture_output=True, text=True,
                timeout=self.timeout, env=env, cwd=tempfile.gettempdir(),  # away from the project's CLAUDE.md
            )
        except FileNotFoundError as exc:
            raise ConfigurationError(f"Claude Code CLI not found at {self.claude_path!r}; install it or set claude_path") from exc
        except subprocess.TimeoutExpired as exc:
            raise LMTimeoutError(f"Claude Code CLI timed out after {self.timeout} s", provider="claude_code") from exc
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            detail = (proc.stderr or proc.stdout).strip()[:500]
            raise ProviderError(f"Claude Code CLI failed (exit {proc.returncode}): {detail}", provider="claude_code") from None
        if data.get("is_error") or proc.returncode != 0:
            raise ProviderError(f"Claude Code CLI returned an error: {str(data.get('result', data))[:500]}", provider="claude_code")
        usage = data.get("usage") or {}
        return data.get("result", ""), usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def stream(self, request):
        return response_to_events(self.complete(request))

    def close(self):
        pass

    # Lets dspy save and reload a program that uses this engine.
    def dump_state(self):
        return {"claude_path": self.claude_path, "timeout": self.timeout}

    @classmethod
    def load_state(cls, state):
        return cls(**state)


# Appended to dspy's system prompt for the CLI. Claude Code's model knows native tool calling, and in
# long ReAct runs it sometimes writes <invoke ...> markup instead of dspy's [[ ## field ## ]] reply.
NO_TOOL_MARKUP = (
    "\n\nYou have no native tools in this conversation. Never write function-call or tool-use markup "
    "such as <invoke> or <function_calls>. When you are asked to choose a tool, write its name and "
    "arguments in the requested [[ ## field ## ]] sections, and reply only in the format described above."
)
TOOL_MARKUP = re.compile(r"<(invoke|function_calls|tool_call|antml:invoke)\b")


def format_problem(prompt, reply):
    """What is wrong with `reply` against dspy's ChatAdapter instructions in `prompt`, or '' if nothing.

    dspy ends its prompt with "Respond with the corresponding output fields, starting with the field
    `[[ ## a ## ]]`, then ..., and then ending with the marker for `[[ ## completed ## ]]`." Every field
    named there must appear in the reply. Prompts without that sentence are not checked."""
    instruction = re.search(r"Respond with the corresponding output fields(.*?)(?:\n\n|$)", prompt, re.S)
    if not instruction:
        return ""
    required = re.findall(r"\[\[ ## (\w+) ## \]\]", instruction.group(1))
    missing = [name for name in required if f"[[ ## {name} ## ]]" not in reply]
    problems = []
    if missing:
        problems.append("it is missing " + ", ".join(f"[[ ## {name} ## ]]" for name in missing))
    if TOOL_MARKUP.search(reply):
        problems.append("it contains tool-call markup, but you have no native tools")
    return " and ".join(problems)


def make_lm(config):
    """The orchestrator's LM from its config: the Anthropic API (backend = api) or the Claude Code CLI."""
    if config.backend == "api":
        return dspy.LM(config.model, max_tokens=config.max_tokens)
    if config.backend == "claude_code":
        # One retry: a timed-out CLI call can be retried, but each attempt may take `timeout` seconds.
        return dspy.LM(config.model, engine=ClaudeCodeEngine(config.claude_path, config.timeout), num_retries=1)
    raise ValueError(f"unknown orchestrator backend {config.backend!r}; use 'api' or 'claude_code'")


def as_steps(value):
    """Claude's `steps` answer as a list of reaction SMILES. dspy normally parses it into a list, but
    some dspy and Python versions hand back the raw text: a JSON list, or one reaction per line."""
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [line.strip().lstrip("-*0123456789.) ").strip() for line in text.splitlines()]
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()]


def parse_reaction(text):
    """'A.B>>P' -> ([canonical A, canonical B], canonical P). Raise ValueError if it cannot be read."""
    left, sep, right = text.partition(">>")
    if not sep:
        raise ValueError(f"not a reaction SMILES (needs '>>'): {text!r}")
    reactants = [canonical(r) for r in left.split(".") if r.strip()]
    product = canonical(right.strip())
    if not reactants or None in reactants or product is None:
        raise ValueError(f"reaction SMILES has an unreadable molecule: {text!r}")
    return reactants, product


def route_from_steps(steps, target):
    """Route tree from reaction SMILES in forward order. Return (root or None, issues).
    A reactant made by an earlier step becomes that step's subtree; any other reactant is a leaf."""
    made, issues = {}, []
    for text in steps:
        try:
            reactants, product = parse_reaction(text)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        children = [made.pop(r) if r in made else TreeNode(r) for r in reactants]
        made[product] = TreeNode(product, 1.0, children)
    root = made.pop(canonical(target), None)
    if root is None:
        issues.append(f"no step makes the target {target}")
    if made:
        issues.append(f"intermediates made but not used: {', '.join(made)}")
    return root, issues


class ProgressCallback(BaseCallback):
    """Turns dspy's ReAct events into progress events, so a caller can show each step as it happens.

    `report(event, data)` is called with:
      "thinking"     {"iteration"}                              Claude is choosing its next action
      "thought"      {"iteration", "thought", "tool", "args"}   Claude's reasoning and the tool it chose
      "tool_start"   {"iteration", "tool", "args"}              a tool started
      "tool_end"     {"iteration", "tool", "result", "seconds"} a tool finished
      "answering"    {}                                         Claude is writing the final answer
    """

    def __init__(self, agent, report):
        self.agent = agent  # the dspy.ReAct instance
        self.report = report
        self.iteration = 0
        self._calls = {}  # call_id -> what started, so the matching end event can be named

    def on_module_start(self, call_id, instance, inputs):
        if instance is self.agent.react:
            self.iteration += 1
            self._calls[call_id] = "react"
            self.report("thinking", {"iteration": self.iteration})
        elif instance is self.agent.extract:
            self.report("answering", {})

    def on_module_end(self, call_id, outputs, exception=None):
        if self._calls.pop(call_id, None) == "react" and outputs is not None:
            self.report("thought", {
                "iteration": self.iteration,
                "thought": outputs.next_thought,
                "tool": outputs.next_tool_name,
                "args": outputs.next_tool_args,
            })

    def on_tool_start(self, call_id, instance, inputs):
        name = getattr(instance, "name", str(instance))
        self._calls[call_id] = (name, time.monotonic())
        self.report("tool_start", {"iteration": self.iteration, "tool": name, "args": inputs})

    def on_tool_end(self, call_id, outputs, exception=None):
        name, started = self._calls.pop(call_id, ("?", time.monotonic()))
        result = f"error: {exception}" if exception is not None else outputs
        self.report("tool_end", {
            "iteration": self.iteration, "tool": name, "result": result, "seconds": time.monotonic() - started,
        })


class RetroOrchestrator(dspy.Module):
    """Claude proposes monomers and steps, validates them with tools, and the result is re-checked.

    `progress`, if given, is called as progress(event, data) while the orchestrator runs: the
    ProgressCallback events, then "recheck_start" {"n_steps"}, "recheck_step" {"index", "step"} and
    "recheck_done" {"valid", "issues"}."""

    def __init__(self, stock, config=None, progress=None):
        super().__init__()
        self.config = config or Config()
        self.stock = stock
        self.progress = progress
        self.validator = RouteValidator(self.config.validator)  # gate and energies; its LLM review is not used
        self.lm = make_lm(self.config.orchestrator)
        self.agent = dspy.ReAct(
            PlanSynthesis,
            tools=[self.validate_step, self.in_stock],
            max_iters=self.config.orchestrator.max_iters,
        )

    def forward(self, target, context=""):
        callbacks = [ProgressCallback(self.agent, self.progress)] if self.progress else []
        with dspy.context(lm=self.lm, callbacks=callbacks):
            pred = self.agent(target=target, context=context)
        result = self.check(target, as_steps(pred.steps))
        result.rationale = pred.rationale
        result.trajectory = dict(pred.trajectory)
        return result

    def check(self, target, steps):
        """Re-check a proposed path without the LLM: gate, energy criteria, and stock, for every step."""
        root, issues = route_from_steps(steps, target)
        checks = []
        if root is not None:
            forward_steps = list(reversed(route_steps(root)))
            self._report("recheck_start", {"n_steps": len(forward_steps)})
            for i, (product, reactants) in enumerate(forward_steps, 1):
                checks.append(self._check_step(product, Counter(reactants)))
                self._report("recheck_step", {"index": i, "step": checks[-1]})
            for s in checks:
                if not s.gate_ok:
                    issues.append(f"{s.product}: gate failed: {s.message}")
                elif not s.deltas:
                    issues.append(f"{s.product}: no energy ({s.note})")
                elif any(v >= 0 for v in s.deltas.values()):
                    issues.append(f"{s.product}: not favourable: {format_deltas(s.deltas)}")
            issues += [f"not in stock: {leaf.smiles}" for leaf in root.leaves() if leaf.smiles not in self.stock]
        valid = root is not None and not issues
        self._report("recheck_done", {"valid": valid, "issues": issues})
        return OrchestratorResult(valid=valid, rationale="", steps=checks, issues=issues, root=root)

    def _report(self, event, data):
        if self.progress:
            self.progress(event, data)

    def _check_step(self, product, reactants):
        """StepCheck for one step; `reactants` is a {smiles: count} dict. Energies only if the gate passes."""
        ok, _, message = check_step_gate(product, reactants)
        step = StepCheck(product, list(reactants.elements()), ok, message)
        if ok:
            try:
                step.deltas = self.validator.step_deltas(product, reactants)
            except ENERGY_ERRORS as exc:
                step.note = str(exc)
        return step

    # Tools called by Claude. They return text, and errors come back as text too,
    # so Claude can read them and revise the step.

    def validate_step(self, product: str, reactants: str) -> str:
        """Validate one step, reactants -> product: atom-balance gate, then the energy changes in kcal/mol.
        product: SMILES, or PSMILES with [*] end points for a polymer repeat unit.
        reactants: comma-separated SMILES, each with an optional ':count', e.g. 'C=C' or 'Nc1ccc(N)cc1, O=C(Cl)c1ccc(C(=O)Cl)cc1'."""
        if canonical(product) is None:
            return f"FAIL: product is not valid SMILES: {product!r}"
        try:
            species = parse_species(reactants)
        except ValueError as exc:
            return f"FAIL: {exc}"
        step = self._check_step(product, Counter(species))
        if not step.gate_ok:
            return f"FAIL: gate failed: {step.message}"
        if not step.deltas:
            return f"FAIL: gate ok ({step.message}), but no energy: {step.note}"
        verdict = "PASS" if all(v < 0 for v in step.deltas.values()) else "FAIL: not favourable"
        text = f"{verdict}: gate ok ({step.message}); {format_deltas(step.deltas)}"
        if step.ceiling_temperature is not None:
            text += f"; ceiling temperature {step.ceiling_temperature:.0f} K"
        return text

    def in_stock(self, smiles: str) -> str:
        """Whether starting materials are purchasable. smiles: comma-separated SMILES."""
        lines = []
        for smi in (s.strip() for s in smiles.split(",") if s.strip()):
            can = canonical(smi)
            status = "invalid SMILES" if can is None else ("in stock" if can in self.stock else "NOT in stock")
            lines.append(f"{smi}: {status}")
        return "\n".join(lines)


class ProgressPrinter:
    """Prints orchestrator progress events as they arrive, one block per Claude iteration."""

    def __init__(self, stream=None, color=None):
        self.stream = stream or sys.stdout
        if color is None:
            color = self.stream.isatty() and "NO_COLOR" not in os.environ
        self.color = color
        self.width = min(shutil.get_terminal_size((100, 20)).columns, 110)
        self.started = time.monotonic()

    def _style(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _print(self, text="", indent=""):
        for line in text.splitlines() or [""]:
            wrapped = textwrap.wrap(line, self.width - len(indent), subsequent_indent="  ") or [""]
            for part in wrapped:
                print(indent + part, file=self.stream, flush=True)

    def header(self, target, config):
        criteria = ", ".join(c.strip() for c in config.validator.criterion.split(","))
        self._print(self._style(f"Target  {target}", "1"))
        via = " via Claude Code" if config.orchestrator.backend == "claude_code" else ""
        self._print(f"Claude  {config.orchestrator.model}{via}  |  energies  {config.validator.backend} ({criteria})"
                    f"  |  up to {config.orchestrator.max_iters} tool calls")

    def __call__(self, event, data):
        clock = self._style(f"{time.monotonic() - self.started:6.1f}s", "2")
        if event == "thinking":
            self._print()
            label = self._style(f"[{data['iteration']}]", "1")
            self._print(f"{clock}  {label} Claude is thinking...")
        elif event == "thought":
            self._print(self._style("Thought: ", "36") + str(data["thought"]), indent="         ")
            if data["tool"] != "finish":
                self._print(f"Next: {data['tool']}({format_args(data['args'])})", indent="         ")
        elif event == "tool_start":
            if data["tool"] == "finish":
                self._print(f"{clock}  Claude is done with the tools.")
            else:
                busy = " (gate, then energies; this can take a while)" if data["tool"] == "validate_step" else ""
                self._print(f"{clock}  -> {data['tool']}{busy}")
        elif event == "tool_end" and data["tool"] != "finish":
            result = str(data["result"])
            code = "32" if result.startswith(("PASS", "in stock")) else ("31" if "FAIL" in result or "NOT" in result else "0")
            self._print(f"{clock}  <- {self._style(result, code)}  ({data['seconds']:.1f} s)")
        elif event == "answering":
            self._print()
            self._print(f"{clock}  Claude is writing its answer...")
        elif event == "recheck_start":
            self._print()
            self._print(self._style(f"Re-checking Claude's {data['n_steps']} step(s) without the LLM", "1"))
        elif event == "recheck_step":
            self._print(format_step(data["index"], data["step"], self._style), indent="  ")
        elif event == "recheck_done":
            verdict = self._style("VALID", "32;1") if data["valid"] else self._style("NOT VALID", "31;1")
            self._print(f"{clock}  Route is {verdict}")


def format_args(args):
    """{'product': 'C=C', 'reactants': 'CC'} -> 'product=C=C, reactants=CC'."""
    if isinstance(args, dict):
        return ", ".join(f"{k}={v}" for k, v in args.items())
    return str(args)


def format_step(index, step, style=lambda text, code: text):
    gate = style("gate ok", "32") if step.gate_ok else style(f"gate FAILED: {step.message}", "31")
    energy = f"  {format_deltas(step.deltas)}" if step.deltas else (f"  no energy: {step.note}" if step.note else "")
    tc = f"  Tc={step.ceiling_temperature:.0f} K" if step.ceiling_temperature is not None else ""
    return f"step {index}: {' + '.join(step.reactants)} -> {step.product}  {gate}{energy}{tc}"


def main():
    parser = argparse.ArgumentParser(description="Claude orchestrator: monomers and validated reaction steps")
    parser.add_argument("target", help="target SMILES, or polymer PSMILES with [*] end points")
    parser.add_argument("--stock", required=True, help="building blocks file, one SMILES per line")
    parser.add_argument("--context", default="", help="extra instructions for Claude")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="settings file (INI)")
    parser.add_argument("--quiet", action="store_true", help="print only the result, not each step as it happens")
    parser.add_argument("--claude-code", action="store_true",
                        help="reach Claude through the local Claude Code CLI (its own login) instead of the API")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.claude_code:
        config = replace(config, orchestrator=replace(config.orchestrator, backend="claude_code"))
    stock = load_stock(args.stock)
    printer = None if args.quiet else ProgressPrinter()
    if printer:
        printer.header(args.target, config)
    try:
        result = RetroOrchestrator(stock, config, progress=printer)(args.target, context=args.context)
    except AdapterParseError as exc:
        reply = str(exc.lm_response or "").strip()
        expected = ", ".join(exc.signature.output_fields) if exc.signature is not None else "the expected fields"
        print(f"\nStopped: Claude's reply could not be read as {expected}.", file=sys.stderr)
        if reply:
            print("Claude's reply was:\n" + textwrap.indent(reply[:2000], "  "), file=sys.stderr)
        print("Run the command again; a long run can drift from the format. A narrower --context can help.", file=sys.stderr)
        sys.exit(1)
    except LMError as exc:
        # A refused or failed LM call (safeguard flag, auth, billing, rate limit, timeout): the
        # provider's message says what happened; the traceback adds nothing for the user.
        print(f"\nStopped: the call to Claude failed.\n{textwrap.indent(str(exc).strip(), '  ')}", file=sys.stderr)
        print("Run the command again, or set a different `model` (or `backend`) in [orchestrator].", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"Route valid={result.valid}")
    for i, s in enumerate(result.steps, 1):
        print("  " + format_step(i, s))
    for issue in result.issues:
        print(f"  ! {issue}")
    print(f"Claude: {result.rationale}")
    if result.root is not None and result.root.children:
        for path in RouteRenderer(config.renderer)(RouteRanker(stock)([result.root])):
            print(f"Image: {path}")


if __name__ == "__main__":
    main()
