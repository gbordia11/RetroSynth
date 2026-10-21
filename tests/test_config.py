import pytest

from config import Config, load_config
from predictor import SingleStepPredictor


def write_ini(tmp_path, text):
    path = tmp_path / "config.ini"
    path.write_text(text)
    return path


def test_shipped_config_loads_with_expected_values():
    config = load_config()
    assert config.predictor.model == "anthropic/claude-sonnet-5-5"
    assert config.predictor.k == 5
    assert config.search.max_depth == 4
    assert config.renderer.out_dir == "retro_routes"


def test_missing_sections_and_keys_use_defaults(tmp_path):
    config = load_config(write_ini(tmp_path, "[search]\nbeam = 7\n"))
    assert config.search.beam == 7
    assert config.search.max_depth == Config().search.max_depth
    assert config.predictor == Config().predictor


def test_each_agent_gets_its_own_model(tmp_path, stock):
    path = write_ini(tmp_path, "[predictor]\nmodel = openai/gpt-4o\nmax_tokens = 1234\n")
    config = load_config(path)
    predictor = SingleStepPredictor(config.predictor)
    assert predictor.lm.model == "openai/gpt-4o"
    assert predictor.lm.kwargs["max_tokens"] == 1234


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown key 'modle'"):
        load_config(write_ini(tmp_path, "[predictor]\nmodle = x\n"))


def test_unknown_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown config section"):
        load_config(write_ini(tmp_path, "[predicter]\nk = 3\n"))


def test_integer_values_are_parsed(tmp_path):
    config = load_config(write_ini(tmp_path, "[renderer]\nwidth = 300\n"))
    assert config.renderer.width == 300


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.ini")
