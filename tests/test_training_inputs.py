import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "train_iaf", Path(__file__).resolve().parents[1] / "scripts/train_iaf.py"
)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


@pytest.mark.parametrize("field", ["structure_log_probabilities", "log_probabilities"])
def test_training_accepts_both_structural_score_fields(tmp_path, field):
    common = dict(patch_id="t:H3:0", target_id="t", region="H3",
                  native_amino_acid="A", native_CDR_tokens_visible=False)
    structure, framework = tmp_path / "s.jsonl", tmp_path / "f.jsonl"
    structure.write_text(json.dumps(dict(common, **{field: [-10.] * 20})) + "\n")
    framework.write_text(json.dumps(dict(common, all_cdr_mask_log_probabilities=[-5.] * 20)) + "\n")
    rows = train.assemble(structure, framework)
    np.testing.assert_allclose(np.exp(rows[0]["zs"]), np.full(20, 0.05))


def test_training_rejects_full_vocabulary_scores():
    with pytest.raises(ValueError, match="20 finite"):
        train.log_softmax(np.zeros(21))
