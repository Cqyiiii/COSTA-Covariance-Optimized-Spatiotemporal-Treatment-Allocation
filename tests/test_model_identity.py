"""Small real simulations protect model identity from CLI through saved results."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from costa.outcomes import OutcomeParameters, generate_observed_outcomes, true_tau

SPEC = importlib.util.spec_from_file_location(
    "run_simulation", Path(__file__).resolve().parents[1] / "scripts/run_simulation.py"
)
sim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sim)
MODELS = ["linear", "nonlinear", "demand", "interaction_second_hop"]


@pytest.mark.parametrize("requested, expected", [
    ("linear,nonlinear,demand,interaction_second_hop", MODELS),
    ("second_hop,demand,interaction_second_hop", ["interaction_second_hop", "demand"]),
    ([" LINEAR ", "nonlinear_saturation", "demand_substitution", "second_hop"], MODELS),
    (("second_hop", "demand"), ["interaction_second_hop", "demand"]),
    ("linear;nonlinear;demand;second_hop", MODELS),
    ("", MODELS[:3]),
])
def test_parse_preserves_model_identity(requested, expected):
    assert sim.parse_models(requested) == expected


def test_defaults_and_unknown_model():
    assert sim.parse_models(sim.build_parser().parse_args([]).models) == MODELS[:3]
    with pytest.raises(ValueError, match="Unknown outcome model"):
        sim.parse_models("not_a_model")


@pytest.mark.parametrize("model, expected", [
    ("linear", [[1.2, 1.7, 1.2], [1.4, 1.2, 1.7]]),
    ("nonlinear", [[1.2, 1.662059301457947, 1.35],
                   [1.362059301457947, 1.2, 1.662059301457947]]),
    ("demand", [[1.2, 1.2, 0.6], [0.6, 0.6, 1.2]]),
    ("interaction_second_hop", [[1.3, 1.7, 1.3], [1.4, 1.3, 1.7]]),
])
@pytest.mark.parametrize("sparse_graph", [False, True])
def test_generated_outcomes_match_hand_computed_models(model, expected, sparse_graph):
    # Two-node swap graph: A^2 = I, so second-hop exposure is Z itself.
    A = np.array([[0., 1.], [1., 0.]])
    if sparse_graph:
        A = sparse.csr_matrix(A)
    Z = np.array([[1, 0, 1], [0, 1, 0]])
    params = OutcomeParameters(
        alpha=np.ones((2, 3)), beta=np.full((2, 3), 0.2),
        gamma=0.4, eta=0.3, model=sim.parse_models(model)[0],
        noise_sd=0, item_shock_sd=0, block_shock_sd=0,
    )
    np.testing.assert_allclose(generate_observed_outcomes(Z, A, params), expected)


@pytest.mark.parametrize("use_config", [False, True])
def test_real_simulation_keeps_parameters_seeds_and_output_labels(tmp_path, monkeypatch, use_config):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    A = sparse.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32))
    weights = np.arange(1., 13.).reshape(3, 4)
    sparse.save_npz(prepared / "competition_A.npz", A)
    np.save(prepared / "item_block_weight.npy", weights)
    out = tmp_path / "out"
    argv = ["run_simulation", "--prepared-dir", str(prepared), "--out-dir", str(out),
            "--designs", "independent", "--mc", "1", "--no-progress", "--no-plot",
            "--second-hop-scale", "0.75", "--interaction-scale", "0.8"]
    if use_config:
        config = tmp_path / "models.yaml"
        config.write_text("models: [linear, nonlinear, demand, second_hop]\n")
        argv += ["--config", str(config)]
    else:
        argv += ["--models", ",".join(MODELS)]
    monkeypatch.setattr(sys, "argv", argv)
    observed = {}
    original = sim.generate_observed_outcomes

    def record_outcomes(Z, graph, params, seed):
        Y = original(Z, graph, params, seed=seed)
        observed[params.model] = (Z.copy(), params, seed, Y)
        return Y

    monkeypatch.setattr(sim, "generate_observed_outcomes", record_outcomes)
    sim.main()
    assert list(observed) == MODELS
    config = json.loads((out / "run_config.json").read_text())
    assert config["models"] == MODELS
    assert config["second_hop_scale"] == 0.75
    assert config["interaction_scale"] == 0.8
    for filename in ("simulation_draws.csv", "simulation_summary.csv",
                     "simulation_summary_primary.csv", "simulation_summary_wide_by_model.csv",
                     "monitor_repetition_diagnostics.csv"):
        frame = pd.read_csv(out / filename)
        assert list(frame.model.unique()) == MODELS
    draws = pd.read_csv(out / "simulation_draws.csv").set_index("model")
    for model_id, model in enumerate(MODELS, 1):
        row = draws.loc[model]
        Z, params, outcome_seed, Y = observed[model]
        assert params.model == model
        assert params.second_hop_scale == 0.75
        assert params.interaction_scale == 0.8
        assert row.model_seed_id == model_id
        for stream_id, stream in enumerate(("assignment", "outcome", "baseline"), 1):
            expected_seed = int(np.random.SeedSequence([123, model_id, 0, stream_id]).generate_state(1)[0])
            assert row[f"{stream}_seed"] == expected_seed
        assert outcome_seed == row.outcome_seed
        np.testing.assert_array_equal(Y, original(Z, A, params, seed=outcome_seed))
        W = weights + 0.05 * weights.mean()
        assert row.tau_true == pytest.approx(true_tau(A, params, weights=W))
        if model == "interaction_second_hop":
            comp = A @ Z
            lag = np.zeros_like(Z)
            lag[:, 1:] = Z[:, :-1]
            expected_mean = (params.alpha + params.beta * Z + params.gamma * comp
                             + 0.8 * params.gamma * Z * comp
                             + 0.75 * params.gamma * (A @ comp) + params.eta * lag)
            from dataclasses import replace
            noiseless = replace(params, noise_sd=0, item_shock_sd=0, block_shock_sd=0)
            np.testing.assert_allclose(original(Z, A, noiseless), expected_mean)
