"""Inference loads what training saves, for the DeepLab and the tessera-embed models."""
import pytest

torch = pytest.importorskip("torch")


def test_deeplab_inference_loads_a_three_class_training_checkpoint(tmp_path):
    from src import inference
    from src.deeplab_model import get_deeplab_model
    from src.ML_farm_net import NUM_CLASSES

    path = tmp_path / "deeplab.pth"
    torch.save(get_deeplab_model(NUM_CLASSES, rgb_from_imagenet=False).state_dict(), path)
    model = inference.load_model(str(path))
    with torch.no_grad():
        assert model(torch.zeros(1, 7, 32, 32))["out"].shape == (1, NUM_CLASSES, 32, 32)


def test_tessera_embed_inference_loads_a_training_checkpoint(tmp_path):
    from src import tessera_embed_infer, tessera_embed_train
    from src.tessera_embed_model import TesseraEmbeddingSegHead

    config = {"in_channels": 128, "num_classes": 4, "hidden_channels": 128}
    path = tmp_path / "embed.pt"
    tessera_embed_train.save_checkpoint(TesseraEmbeddingSegHead(**config), str(path), config, {})
    model = tessera_embed_infer.load_model(str(path))
    with torch.no_grad():
        assert model(torch.zeros(1, 128, 8, 8))["out"].shape == (1, 4, 8, 8)
