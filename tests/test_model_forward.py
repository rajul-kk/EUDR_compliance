"""Tests for SiameseDeepLabV3 forward pass shape."""
import pytest

torch = pytest.importorskip("torch")


def test_siamese_forward_output_shape():
    from src.change_siamese_model import get_siamese_model
    model = get_siamese_model()
    model.eval()
    t1 = torch.randn(1, 6, 64, 64)
    t2 = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        out = model(t1, t2)
    assert "out" in out
    assert out["out"].shape == (1, 2, 64, 64)


def test_siamese_forward_two_classes():
    from src.change_siamese_model import get_siamese_model
    model = get_siamese_model()
    model.eval()
    t1 = torch.zeros(1, 6, 32, 32)
    t2 = torch.zeros(1, 6, 32, 32)
    with torch.no_grad():
        out = model(t1, t2)
    assert out["out"].shape[1] == 2
