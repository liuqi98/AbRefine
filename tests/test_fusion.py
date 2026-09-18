import json

import numpy as np
import pytest

from gatefree_antibody_fusion.fusion import fuse_log_probabilities
from gatefree_antibody_fusion.io import validate_pair
from gatefree_antibody_fusion.metrics import paired_comparison, target_macro


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

