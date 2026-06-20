from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MVTecDataset
from .features import RandomProjector, WideResNetPatchExtractor, extract_loader
from .metrics import evaluate
from .models import PatchMemory, PlasticPrototypeMemory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class ExperimentRunner:
    def __init__(self, config: dict, run_name: str) -> None:
        self.cfg = config
        self.device = resolve_device(config["device"])
        self.out = Path(config["output_root"]) / run_name
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "heatmaps").mkdir(exist_ok=True)
        with open(self.out / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        self.extractor = WideResNetPatchExtractor(config["features"]["layers"]).to(self.device)
        self.rows: list[dict] = []

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset, batch_size=self.cfg["batch_size"], shuffle=shuffle,
            num_workers=self.cfg["num_workers"], pin_memory=self.device.type == "cuda",
        )

    def _features(self, category: str, shot: int, seed: int, drift_level: float = 0.0):
        train = MVTecDataset(
            self.cfg["data_root"], category, "train", self.cfg["image_size"], shot, seed,
        )
        test = MVTecDataset(
            self.cfg["data_root"], category, "test", self.cfg["image_size"],
            drift_level=drift_level,
        )
        train_data = extract_loader(self._loader(train), self.extractor, self.device)
        test_data = extract_loader(self._loader(test), self.extractor, self.device)
        n, h, w, d = train_data["features"].shape
        projector = RandomProjector(
            self.cfg["features"]["projection_dim"], seed,
            self.cfg["features"]["projection_fit_patches"],
        ).fit(train_data["features"].reshape(-1, d))
        train_data["features"] = projector.transform(train_data["features"].reshape(-1, d)).reshape(n, h, w, -1)
        nt, ht, wt, dt = test_data["features"].shape
        test_data["features"] = projector.transform(test_data["features"].reshape(-1, dt)).reshape(nt, ht, wt, -1)
        return train_data, test_data, projector

    def _ppm(self, seed: int, **overrides) -> PlasticPrototypeMemory:
        cfg = dict(self.cfg["ppm"])
        cfg.update(overrides)
        return PlasticPrototypeMemory(
            n_prototypes=cfg["prototypes"], radius_quantile=cfg["radius_quantile"],
            temperature=cfg["temperature"], attractor_steps=cfg["attractor_steps"],
            attractor_lr=cfg["attractor_lr"], group_beta=cfg["group_beta"],
            group_sigma_feature=cfg["group_sigma_feature"], score_weights=cfg["score_weights"],
            plasticity_lr=cfg["plasticity_lr"], max_update_ratio=cfg["max_update_ratio"],
            seed=seed, device=self.device,
        )

    @staticmethod
    def _upsample(score_maps: list[np.ndarray], target_hw: tuple[int, int]) -> np.ndarray:
        values = torch.as_tensor(np.asarray(score_maps), dtype=torch.float32)[:, None]
        return F.interpolate(values, target_hw, mode="bilinear", align_corners=False)[:, 0].numpy()

    @staticmethod
    def _image_scores(pixel_scores: np.ndarray, ratio: float = 0.01) -> np.ndarray:
        flat = pixel_scores.reshape(len(pixel_scores), -1)
        count = max(1, int(flat.shape[1] * ratio))
        return np.partition(flat, -count, axis=1)[:, -count:].mean(1)

    def _evaluate_ppm(self, model, test_data, variant: str):
        enabled, attractor, group = None, True, True
        if variant == "prototype-only":
            enabled, attractor, group = {"distance"}, False, False
        elif variant == "w/o competition":
            enabled = {"distance", "residual", "stability"}
        elif variant == "w/o attractor":
            enabled, attractor = {"distance", "entropy", "margin"}, False
        elif variant == "w/o group":
            group = False
        score_maps, inference = [], []
        for fmap in test_data["features"]:
            started = time.perf_counter()
            score, _ = model.score(fmap, enabled=enabled, use_attractor=attractor, use_group=group)
            inference.append(time.perf_counter() - started)
            score_maps.append(score)
        pixel_scores = self._upsample(score_maps, test_data["masks"].shape[-2:])
        return pixel_scores, float(np.mean(inference) * 1000)

    def _evaluate_patch(self, model, test_data):
        score_maps, inference = [], []
        for fmap in test_data["features"]:
            h, w, d = fmap.shape
            started = time.perf_counter()
            score = model.score(fmap.reshape(-1, d)).reshape(h, w)
            inference.append(time.perf_counter() - started)
            score_maps.append(score)
        pixel_scores = self._upsample(score_maps, test_data["masks"].shape[-2:])
        return pixel_scores, float(np.mean(inference) * 1000)

    def _record(self, metadata: dict, test_data: dict, pixel_scores: np.ndarray, fit_ms: float, infer_ms: float, memory_bytes: int):
        image_scores = self._image_scores(pixel_scores)
        row = {
            **metadata,
            **evaluate(test_data["labels"], image_scores, test_data["masks"], pixel_scores),
            "fit_ms": fit_ms,
            "inference_ms_per_image": infer_ms,
            "memory_mb": memory_bytes / (1024 ** 2),
            "normal_mean_score": float(image_scores[test_data["labels"] == 0].mean()),
            "anomaly_mean_score": float(image_scores[test_data["labels"] == 1].mean()),
        }
        self.rows.append(row)
        self._flush()
        return image_scores

    def _flush(self):
        frame = pd.DataFrame(self.rows)
        frame.to_csv(self.out / "results_raw.csv", index=False)
        if len(frame):
            metric_columns = [
                "image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro_0.3",
                "max_f1", "iou_at_best_f1", "fragmentation", "fit_ms",
                "inference_ms_per_image", "memory_mb",
            ]
            group_columns = [x for x in ("experiment", "model", "shot", "category", "drift_level", "policy", "k") if x in frame.columns]
            summary = frame.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std"]).reset_index()
            summary.to_csv(self.out / "results_summary.csv", index=False)
        with open(self.out / "run_manifest.json", "w", encoding="utf-8") as handle:
            json.dump({
                "device": str(self.device), "torch": torch.__version__,
                "cuda": torch.version.cuda, "rows_completed": len(self.rows),
            }, handle, ensure_ascii=False, indent=2)

    def _save_heatmaps(self, category, model_name, test_data, scores, limit):
        anomaly_indices = np.flatnonzero(test_data["labels"] == 1)[:limit]
        for index in anomaly_indices:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            axes[0].imshow(test_data["masks"][index, 0], cmap="gray")
            axes[0].set_title("Ground truth")
            axes[1].imshow(scores[index], cmap="inferno")
            axes[1].set_title(f"{model_name} anomaly map")
            for axis in axes:
                axis.axis("off")
            fig.tight_layout()
            path = self.out / "heatmaps" / f"{category}_{model_name.replace(' ', '_')}_{index}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)

    def run_fewshot(self):
        for category in self.cfg["categories"]:
            for shot in self.cfg["shots"]:
                for seed in self.cfg["seeds"]:
                    set_seed(seed)
                    train, test, _ = self._features(category, shot, seed)
                    patches = train["features"].reshape(-1, train["features"].shape[-1])
                    models = [
                        ("patchcore", PatchMemory(self.cfg["patchcore"]["max_memory_patches"], seed, self.device), 1.0),
                        ("patchcore-coreset", PatchMemory(self.cfg["patchcore"]["max_memory_patches"], seed, self.device), self.cfg["patchcore"]["coreset_ratio"]),
                    ]
                    for name, model, ratio in models:
                        started = time.perf_counter(); model.fit(patches, ratio); fit_ms = (time.perf_counter() - started) * 1000
                        scores, infer_ms = self._evaluate_patch(model, test)
                        self._record({"experiment": "fewshot", "model": name, "category": category, "shot": shot, "seed": seed}, test, scores, fit_ms, infer_ms, model.memory_bytes)
                    for name in ("prototype-only", "ppm-ad-full"):
                        overrides = {"attractor_steps": 0, "group_beta": 0.0} if name == "prototype-only" else {}
                        model = self._ppm(seed, **overrides).fit(patches)
                        scores, infer_ms = self._evaluate_ppm(model, test, name)
                        self._record({"experiment": "fewshot", "model": name, "category": category, "shot": shot, "seed": seed}, test, scores, model.fit_seconds * 1000, infer_ms, model.memory_bytes)
                        if name == "ppm-ad-full" and self.cfg["experiments"]["save_heatmaps"] and seed == self.cfg["seeds"][0]:
                            self._save_heatmaps(category, name, test, scores, self.cfg["experiments"]["heatmaps_per_category"])

    def run_ablation(self):
        shot = self.cfg["experiments"]["ablation_shot"]
        for category in self.cfg["categories"]:
            for seed in self.cfg["seeds"]:
                train, test, _ = self._features(category, shot, seed)
                patches = train["features"].reshape(-1, train["features"].shape[-1])
                variants = ("ppm-ad-full", "prototype-only", "w/o competition", "w/o attractor", "w/o group")
                for variant in variants:
                    overrides = {}
                    if variant == "prototype-only": overrides = {"attractor_steps": 0, "group_beta": 0.0}
                    if variant == "w/o attractor": overrides = {"attractor_steps": 0}
                    model = self._ppm(seed, **overrides).fit(patches)
                    scores, infer_ms = self._evaluate_ppm(model, test, variant)
                    self._record({"experiment": "ablation", "model": variant, "category": category, "shot": shot, "seed": seed}, test, scores, model.fit_seconds * 1000, infer_ms, model.memory_bytes)

    def run_k_sweep(self):
        shot = self.cfg["experiments"]["ablation_shot"]
        for category in self.cfg["categories"]:
            for seed in self.cfg["seeds"]:
                train, test, _ = self._features(category, shot, seed)
                patches = train["features"].reshape(-1, train["features"].shape[-1])
                for k in self.cfg["experiments"]["k_values"]:
                    model = self._ppm(seed, prototypes=k).fit(patches)
                    scores, infer_ms = self._evaluate_ppm(model, test, "ppm-ad-full")
                    self._record({"experiment": "k_sweep", "model": "ppm-ad-full", "category": category, "shot": shot, "seed": seed, "k": k}, test, scores, model.fit_seconds * 1000, infer_ms, model.memory_bytes)

    def run_drift(self):
        shot = self.cfg["experiments"]["drift_shot"]
        for category in self.cfg["categories"]:
            for seed in self.cfg["seeds"]:
                train, clean_test, projector = self._features(category, shot, seed, 0.0)
                patches = train["features"].reshape(-1, train["features"].shape[-1])
                base = self._ppm(seed).fit(patches)
                policies = {name: base.clone() for name in ("fixed", "ungated", "gated")}
                initial_anomaly = {}
                for level in self.cfg["experiments"]["drift_levels"]:
                    drift_ds = MVTecDataset(self.cfg["data_root"], category, "test", self.cfg["image_size"], drift_level=level)
                    drift = extract_loader(self._loader(drift_ds), self.extractor, self.device)
                    n, h, w, d = drift["features"].shape
                    drift["features"] = projector.transform(drift["features"].reshape(-1, d)).reshape(n, h, w, -1)
                    for policy, model in policies.items():
                        scores, infer_ms = self._evaluate_ppm(model, drift, "ppm-ad-full")
                        image_scores = self._record({"experiment": "drift", "model": "ppm-ad-full", "category": category, "shot": shot, "seed": seed, "drift_level": level, "policy": policy}, drift, scores, model.fit_seconds * 1000, infer_ms, model.memory_bytes)
                        anomaly_mean = float(image_scores[drift["labels"] == 1].mean())
                        initial_anomaly.setdefault(policy, anomaly_mean)
                        self.rows[-1]["delta_anomaly_score"] = initial_anomaly[policy] - anomaly_mean
                        accepted = 0
                        for fmap in drift["features"]:
                            accepted += model.update(fmap.reshape(-1, fmap.shape[-1]), policy)
                        self.rows[-1]["accepted_updates"] = accepted
                        self._flush()


def run(config_path: str, mode: str, run_name: str) -> Path:
    runner = ExperimentRunner(load_config(config_path), run_name)
    modes = {
        "fewshot": runner.run_fewshot,
        "ablation": runner.run_ablation,
        "drift": runner.run_drift,
        "k_sweep": runner.run_k_sweep,
    }
    selected = modes.keys() if mode == "all" else (mode,)
    for name in selected:
        tqdm.write(f"Running {name} on {runner.device}")
        modes[name]()
    return runner.out

