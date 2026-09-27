import json

import numpy as np
import pytest

from abrefine.fusion import fuse_log_probabilities
from abrefine.iaf import IAFEnsemble, IAFMember, information_features
from abrefine.io import validate_pair
from abrefine.metrics import paired_comparison, target_macro


def test_product_of_experts_has_exact_endpoints():
    structure = np.log([0.70, 0.20, 0.10] + [1e-12] * 17)
    evolution = np.log([0.10, 0.30, 0.60] + [1e-12] * 17)
    np.testing.assert_allclose(fuse_log_probabilities(structure, evolution, 0), np.exp(structure) / np.exp(structure).sum())
    np.testing.assert_allclose(fuse_log_probabilities(structure, evolution, 1), np.exp(evolution) / np.exp(evolution).sum())


def test_target_macro_treats_antibodies_as_independent_units():
    rows = [{"target_id": "a", "x": 1}, {"target_id": "a", "x": 1}, {"target_id": "b", "x": 0}]
    assert target_macro(rows, "x") == 0.5


def test_leakage_contract_is_enforced():
    base = {"p": {"target_id": "t", "region": "H3", "native_CDR_tokens_visible": False}}
    prior = {"p": {"target_id": "t", "region": "H3", "native_CDR_tokens_visible": True}}
    with pytest.raises(ValueError, match="leakage"):
        validate_pair(base, prior)


def test_paired_bootstrap_reports_direction():
    rows = [
        {"target_id": "a", "base": 0, "fused": 1},
        {"target_id": "b", "base": 0, "fused": 1},
        {"target_id": "c", "base": 1, "fused": 1},
    ]
    result = paired_comparison(rows, "fused", "base", repeats=1000, seed=7)
    assert result["target_macro_difference"] > 0
    assert result["improved_targets"] == 2


def test_iaf_features_are_finite_and_nonnegative():
    structure = np.log(np.asarray([0.70, 0.20, 0.10] + [1e-12] * 17))
    framework = np.log(np.asarray([0.10, 0.30, 0.60] + [1e-12] * 17))
    empirical = np.full(20, 0.05)
    features = information_features(structure, framework, empirical)
    assert features.shape == (4,)
    assert np.all(np.isfinite(features))
    assert np.all(features >= -1e-12)


def test_zero_network_recovers_anchor_product_of_experts():
    member = IAFMember(
        seed=41,
        mean=np.zeros(4),
        scale=np.ones(4),
        weight1=np.zeros((8, 4)),
        bias1=np.zeros(8),
        weight2=np.zeros(8),
        bias2=0.0,
        g0=0.59,
    )
    model = IAFEnsemble([member])
    structure = np.log(np.asarray([0.70, 0.20, 0.10] + [1e-12] * 17))
    framework = np.log(np.asarray([0.10, 0.30, 0.60] + [1e-12] * 17))
    probability, gate, features = model.refine(structure, framework, np.full(20, 0.05))
    np.testing.assert_allclose(probability, fuse_log_probabilities(structure, framework, 0.59))
    assert gate == pytest.approx(0.59)
    assert features.shape == (4,)


def test_iaf_batch_and_single_inference_agree():
    member = IAFMember(
        seed=7,
        mean=np.zeros(4),
        scale=np.ones(4),
        weight1=np.ones((8, 4)) * 0.01,
        bias1=np.zeros(8),
        weight2=np.ones(8) * 0.02,
        bias2=-0.1,
    )
    model = IAFEnsemble([member])
    structure = np.log(np.asarray([0.70, 0.20, 0.10] + [1e-12] * 17))
    framework = np.log(np.asarray([0.10, 0.30, 0.60] + [1e-12] * 17))
    empirical = np.full(20, 0.05)
    single = model.refine(structure, framework, empirical)
    batch = model.refine(np.stack([structure]), np.stack([framework]), np.stack([empirical]))
    np.testing.assert_allclose(single[0], batch[0][0])
    assert single[1] == pytest.approx(batch[1][0])
