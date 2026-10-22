"""Load pipeline settings from an INI file: one [section] per agent, plain `key = value` lines."""

import configparser
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.ini"

# API keys (e.g. MISTRAL_API_KEY) come from .env. Variables already set in the shell win.
load_dotenv(PROJECT_DIR / ".env", override=False)


@dataclass
class PredictorConfig:
    """Agent 1: LLM-backed single-step predictor."""

    model: str = "anthropic/claude-sonnet-5-5"  # any dspy model string, e.g. openai/gpt-4o
    max_tokens: int = 4000
    k: int = 5  # disconnections returned per molecule


@dataclass
class SearchConfig:
    """Agent 2: retrosynthetic tree search (uses the predictor's model)."""

    max_depth: int = 4
    beam: int = 3
    max_routes: int = 10


@dataclass
class RendererConfig:
    """Agent 4: RDKit route drawing."""

    out_dir: str = "retro_routes"
    width: int = 250
    height: int = 200


@dataclass
class ValidatorConfig:
    """Agent 5: route validator. An LLM calls tools (repeat-unit check, mass balance, ORCA)."""

    enabled: bool = False  # runs the thermodynamic check on every route when True
    backend: str = "gfn2"  # energy source: "gfn2" (GFN2-xTB via tblite) or "orca"
    criterion: str = "dg"  # step must have dG < 0 ("dg", needs gfn2) or dE < 0 ("de")
    model: str = "mistral/open-mistral-nemo"  # reads MISTRAL_API_KEY from .env
    max_tokens: int = 4000
    max_iters: int = 6  # tool-calling steps before the agent must answer
    orca_path: str = "orca"  # binary name on PATH, or full path, used only when backend = orca
    method: str = "B3LYP D3BJ def2-SVP Opt TightSCF"  # ORCA keyword line
    timeout: int = 900  # seconds per ORCA run
    workdir: str = "orca_runs"


@dataclass
class Config:
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)


SECTIONS = {f.name: f.type for f in fields(Config)}


def _build(section_cls, parser, name):
    """Build one section dataclass, rejecting unknown keys so typos fail loudly."""
    known = {f.name: f.type for f in fields(section_cls)}
    values = {}
    if parser.has_section(name):
        for key, raw in parser.items(name):
            if key not in known:
                raise ValueError(f"[{name}] unknown key '{key}'; expected one of {sorted(known)}")
            # bool("false") is True, so booleans need the INI-aware parser.
            values[key] = parser.getboolean(name, key) if known[key] is bool else known[key](raw)
    return section_cls(**values)


def load_config(path=DEFAULT_CONFIG_PATH):
    """Read the INI file at `path`. Missing sections or keys fall back to the defaults."""
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"config file not found: {path}")
    unknown = set(parser.sections()) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown config section(s): {sorted(unknown)}; expected {sorted(SECTIONS)}")
    return Config(**{name: _build(cls, parser, name) for name, cls in SECTIONS.items()})
