import numpy as np

from ppm_ad.metrics import evaluate
from ppm_ad.models import PatchMemory, PlasticPrototypeMemory


def synthetic_features(seed=0):
    rng = np.random.default_rng(seed)
    normal = np.concatenate([
        rng.normal(-0.5, 0.08, (200, 8)),
        rng.normal(0.5, 0.08, (200, 8)),
    ]).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    anomaly = rng.normal(0.0, 0.5, (64, 8)).astype(np.float32)
    anomaly /= np.linalg.norm(anomaly, axis=1, keepdims=True)
    return normal, anomaly


def test_patch_memory_separates_synthetic_anomalies():
    normal, anomaly = synthetic_features()
    model = PatchMemory(max_patches=400).fit(normal)
    assert model.score(anomaly).mean() > model.score(normal[:64]).mean()


def test_ppm_score_update_and_metrics_smoke():
    normal, anomaly = synthetic_features()
    model = PlasticPrototypeMemory(n_prototypes=8, attractor_steps=2, seed=0).fit(normal)
    fmap = anomaly.reshape(8, 8, 8)
    score, components = model.score(fmap)
    assert score.shape == (8, 8)
    assert set(components) == {"distance", "entropy", "margin", "residual", "stability"}
    before = model.centers.clone()
    model.update(normal[:64], policy="gated")
    assert model.centers.shape == before.shape

    masks = np.zeros((2, 1, 8, 8), dtype=np.float32)
    masks[1, 0, 2:6, 2:6] = 1
    pixel_scores = np.stack([np.zeros((8, 8)), masks[1, 0]])
    result = evaluate(np.array([0, 1]), np.array([0.0, 1.0]), masks, pixel_scores)
    assert result["image_auroc"] == 1.0


