from agent import RetroSynth, format_route
from config import Config, PredictorConfig, RendererConfig, SearchConfig
from molecules import TreeNode, canonical, element_counts, load_stock, route_score
from predictor import SingleStepPredictor
from ranker import RouteRanker
from renderer import RouteRenderer
from search import RetroSearchAgent

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC_ACID = "OC(=O)c1ccccc1O"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"
ACETIC_ACID = "CC(=O)O"


# --- molecules.py ---------------------------------------------------------

def test_canonical_returns_none_for_invalid_smiles():
    assert canonical("not_a_smiles") is None
    assert canonical("") is None


def test_canonical_is_stable_for_equivalent_smiles():
    assert canonical("OC(=O)c1ccccc1O") == canonical("O=C(O)c1ccccc1O")


def test_load_stock_reads_first_field_and_skips_header(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("smiles name\nCC(=O)O acetic_acid\n\nbogus_line\n")
    assert load_stock(path) == {canonical("CC(=O)O")}


def test_element_counts_counts_heavy_atoms():
    assert element_counts(ACETIC_ACID) == {"C": 2, "O": 2}


def test_route_score_is_product_of_step_probabilities():
    root = TreeNode("A", 0.5, [TreeNode("B", 0.4, [TreeNode("C")]), TreeNode("D")])
    assert route_score(root) == 0.2


# --- predictor.py ---------------------------------------------------------

def test_predictor_filters_invalid_trivial_and_duplicate(aspirin_table):
    predictor = SingleStepPredictor(PredictorConfig(k=5), backend=lambda s: aspirin_table.get(s, []))
    out = predictor(ASPIRIN)
    assert len(out) == 2
    assert [round(d.prob, 2) for d in out] == [0.8, 0.6]
    assert all(canonical(r) for d in out for r in d.reactants)


def test_predictor_respects_k(aspirin_table):
    predictor = SingleStepPredictor(PredictorConfig(k=1), backend=lambda s: aspirin_table.get(s, []))
    out = predictor(ASPIRIN)
    assert len(out) == 1
    assert out[0].prob == 0.8


def test_predictor_clips_probabilities():
    predictor = SingleStepPredictor(PredictorConfig(k=5), backend=lambda s: [(["CC(=O)O"], 1.7)])
    assert predictor(ASPIRIN)[0].prob == 1.0


# --- search.py ------------------------------------------------------------

def test_search_finds_one_step_routes(stock, aspirin_table):
    predictor = SingleStepPredictor(PredictorConfig(k=5), backend=lambda s: aspirin_table.get(s, []))
    search = RetroSearchAgent(predictor, stock)
    routes = search(ASPIRIN)
    assert [round(route_score(r), 2) for r in routes] == [0.8, 0.6]
    for route in routes:
        assert all(leaf.smiles in stock for leaf in route.leaves())


def test_search_returns_nothing_for_stock_target(stock):
    predictor = SingleStepPredictor(PredictorConfig(k=5), backend=lambda s: [])
    search = RetroSearchAgent(predictor, stock)
    assert search(SALICYLIC_ACID) == []


def test_search_expands_multiple_steps_and_avoids_cycles():
    stock = {canonical("CC"), canonical("O"), canonical("C=O")}
    table = {
        "CCCCO": [(["CCCO", "C=O"], 0.9), (["CCCCO", "C=O"], 0.99)],  # second is a cycle
        "CCCO": [(["CC", "O"], 0.5)],
    }
    predictor = SingleStepPredictor(PredictorConfig(k=5), backend=lambda s: table.get(s, []))
    routes = RetroSearchAgent(predictor, stock, SearchConfig(max_depth=3))("CCCCO")
    assert len(routes) == 1
    assert route_score(routes[0]) == 0.45
    assert len(format_route(routes[0])) == 2


def test_search_respects_max_depth():
    stock = {canonical("CC")}
    table = {"CCCC": [(["CCC"], 0.9)], "CCC": [(["CC"], 0.9)]}
    predictor = SingleStepPredictor(PredictorConfig(k=5), backend=lambda s: table.get(s, []))
    assert RetroSearchAgent(predictor, stock, SearchConfig(max_depth=1))("CCCC") == []
    assert len(RetroSearchAgent(predictor, stock, SearchConfig(max_depth=2))("CCCC")) == 1


# --- ranker.py ------------------------------------------------------------

def test_ranker_accepts_valid_route(stock):
    root = TreeNode(canonical(ASPIRIN), 0.8, [
        TreeNode(canonical(ACETIC_ANHYDRIDE)),
        TreeNode(canonical(SALICYLIC_ACID)),
    ])
    (ranked,) = RouteRanker(stock)([root])
    assert ranked.valid
    assert ranked.issues == []
    assert ranked.n_steps == 1


def test_ranker_flags_missing_elements_and_non_stock_leaves(stock):
    bad = TreeNode(canonical(ASPIRIN), 0.7, [
        TreeNode(canonical(SALICYLIC_ACID)),
        TreeNode("CCCC"),  # not in stock, and contributes no oxygen
    ])
    (ranked,) = RouteRanker(stock)([bad])
    assert not ranked.valid
    assert any("precursors lack" in issue for issue in ranked.issues)
    assert any("leaf not in stock" in issue for issue in ranked.issues)


def test_ranker_puts_valid_routes_first(stock):
    valid = TreeNode(canonical(ASPIRIN), 0.5, [TreeNode(canonical(ACETIC_ANHYDRIDE)),
                                              TreeNode(canonical(SALICYLIC_ACID))])
    invalid = TreeNode(canonical(ASPIRIN), 0.99, [TreeNode("CCCC")])
    ranked = RouteRanker(stock)([invalid, valid])
    assert ranked[0].valid and not ranked[1].valid


# --- renderer.py ----------------------------------------------------------

def test_renderer_writes_one_png_per_route(stock, tmp_path):
    root = TreeNode(canonical(ASPIRIN), 0.8, [
        TreeNode(canonical(ACETIC_ANHYDRIDE)),
        TreeNode(canonical(SALICYLIC_ACID)),
    ])
    ranked = RouteRanker(stock)([root])
    paths = RouteRenderer(RendererConfig(out_dir=str(tmp_path)))(ranked)
    assert [p.name for p in paths] == ["route_01.png"]
    assert paths[0].stat().st_size > 0


# --- agent.py (end to end) ------------------------------------------------

def test_pipeline_end_to_end(stock, aspirin_table, tmp_path):
    config = Config(renderer=RendererConfig(out_dir=str(tmp_path)))
    pipeline = RetroSynth(stock, config, backend=lambda s: aspirin_table.get(s, []))
    result = pipeline("", ASPIRIN)
    assert len(result.routes) == 2
    assert all(r.valid for r in result.routes)
    assert len(result.images) == 2
    assert all(p.exists() for p in result.images)


def test_pipeline_reports_no_route(stock, tmp_path):
    config = Config(renderer=RendererConfig(out_dir=str(tmp_path)))
    pipeline = RetroSynth(stock, config, backend=lambda s: [])
    result = pipeline("", ASPIRIN)
    assert result.routes == []
    assert result.images == []
