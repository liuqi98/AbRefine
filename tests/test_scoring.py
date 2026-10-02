import numpy as np
import pytest

from abrefine.fusion import fuse_log_probabilities
from abrefine.iaf import information_features
from abrefine.metrics import native_residue_nll


def test_full_vocabulary_mass_is_removed_before_scoring():
    full = np.array([0.4] + [0.4 / 19] * 19 + [0.2])
    nll = native_residue_nll(np.log(full[:20]), "A")
    assert nll == pytest.approx(-np.log(0.5))
    assert -np.log(full[0]) == pytest.approx(nll - np.log(0.8))


def test_information_features_ignore_removed_vocabulary_mass():
    zs, zf = np.linspace(-4, 1, 20), np.linspace(2, -3, 20)
    prior = np.full(20, 0.05)
    np.testing.assert_allclose(
        information_features(zs, zf, prior),
        information_features(zs - 5, zf - 9, prior), atol=1e-12,
    )


@pytest.mark.parametrize("weight", [0, 0.59, 1])
def test_extreme_finite_scores_do_not_break_fusion(weight):
    zs = np.array([0.] + [-1000.] * 19)
    zf = zs[::-1]
    result = fuse_log_probabilities(zs, zf, weight)
    assert np.all(np.isfinite(result))
    assert result.sum() == pytest.approx(1)
    assert native_residue_nll(zs, "Y") == pytest.approx(1000)


@pytest.mark.parametrize("values", [np.zeros(21), np.full(20, np.nan)])
def test_nll_rejects_invalid_alphabet_or_scores(values):
    with pytest.raises(ValueError):
        native_residue_nll(values, "A")
