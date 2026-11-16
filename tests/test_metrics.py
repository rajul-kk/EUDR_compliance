"""Tests for train_utils.change_f1 (foreground F1 on change logits, 255 ignored)."""
import pytest

torch = pytest.importorskip("torch")

IGNORE = 255


def _logits(preds):
    """Convert flat predicted class list to (1, 2, H, W) logit tensor."""
    n = len(preds)
    side = int(n ** 0.5)
    log = torch.zeros(1, 2, side, side)
    for i, p in enumerate(preds):
        r, c = divmod(i, side)
        log[0, p, r, c] = 10.0
    return log


def test_perfect_prediction():
    from src.train_utils import change_f1
    targets = torch.tensor([[0, 1], [0, 1]], dtype=torch.long).unsqueeze(0)
    logits = _logits([0, 1, 0, 1])
    assert change_f1(logits, targets) == pytest.approx(1.0, abs=1e-4)


def test_all_wrong():
    from src.train_utils import change_f1
    targets = torch.ones(1, 2, 2, dtype=torch.long)
    logits = _logits([0, 0, 0, 0])
    assert change_f1(logits, targets) == pytest.approx(0.0, abs=1e-4)


def test_all_ignore():
    from src.train_utils import change_f1
    targets = torch.full((1, 2, 2), IGNORE, dtype=torch.long)
    logits = _logits([1, 1, 1, 1])
    assert change_f1(logits, targets) < 1e-3
