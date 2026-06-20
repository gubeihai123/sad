from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


def _device(value: str | torch.device) -> torch.device:
    return value if isinstance(value, torch.device) else torch.device(value)


class PatchMemory:
    """PatchCore-style nearest-neighbour memory baseline."""

    def __init__(self, max_patches: int = 50000, seed: int = 0, device: str = "cpu") -> None:
        self.max_patches = max_patches
        self.seed = seed
        self.device = _device(device)

    def fit(self, patches: np.ndarray, coreset_ratio: float = 1.0) -> "PatchMemory":
        rng = np.random.default_rng(self.seed)
        target = min(len(patches), self.max_patches, max(1, int(len(patches) * coreset_ratio)))
        if target < len(patches):
            # A deterministic random coreset is used as a transparent memory-compression baseline.
            patches = patches[rng.choice(len(patches), target, replace=False)]
        self.memory = torch.as_tensor(patches, dtype=torch.float32, device=self.device)
        return self

    @torch.inference_mode()
    def score(self, patches: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        query = torch.as_tensor(patches, dtype=torch.float32, device=self.device)
        result = []
        for chunk in query.split(chunk_size):
            result.append(torch.cdist(chunk, self.memory).amin(dim=1).cpu())
        return torch.cat(result).numpy()

    @property
    def memory_bytes(self) -> int:
        return self.memory.numel() * self.memory.element_size()


@dataclass
class PlasticPrototypeMemory:
    n_prototypes: int = 256
    radius_quantile: float = 0.95
    temperature: float = 0.2
    attractor_steps: int = 3
    attractor_lr: float = 0.25
    group_beta: float = 0.25
    group_sigma_feature: float = 0.5
    score_weights: Mapping[str, float] = field(default_factory=lambda: {
        "distance": 1.0, "entropy": 0.25, "margin": 0.25,
        "residual": 0.15, "stability": 0.15,
    })
    plasticity_lr: float = 0.005
    max_update_ratio: float = 0.05
    seed: int = 0
    device: str | torch.device = "cpu"

    component_names = ("distance", "entropy", "margin", "residual", "stability")

    def fit(self, patches: np.ndarray) -> "PlasticPrototypeMemory":
        started = time.perf_counter()
        if len(patches) < 2:
            raise ValueError("At least two normal patches are required")
        k = min(self.n_prototypes, len(patches))
        km = MiniBatchKMeans(
            n_clusters=k, batch_size=min(4096, len(patches)), n_init=3,
            random_state=self.seed, reassignment_ratio=0.01,
        ).fit(patches)
        centers = km.cluster_centers_.astype(np.float32)
        labels = km.predict(patches)
        distances = np.linalg.norm(patches - centers[labels], axis=1)
        global_radius = max(float(np.quantile(distances, self.radius_quantile)), 1e-3)
        radii = np.empty(k, dtype=np.float32)
        for idx in range(k):
            local = distances[labels == idx]
            radii[idx] = max(
                float(np.quantile(local, self.radius_quantile)) if len(local) >= 2 else global_radius,
                global_radius * 0.1,
                1e-3,
            )
        self.device = _device(self.device)
        self.centers = torch.as_tensor(centers, device=self.device)
        self.radii = torch.as_tensor(radii, device=self.device)

        # Normal-reference statistics make heterogeneous score terms comparable.
        rng = np.random.default_rng(self.seed)
        reference = patches
        if len(reference) > 20000:
            reference = reference[rng.choice(len(reference), 20000, replace=False)]
        raw, _ = self._raw_components(torch.as_tensor(reference, dtype=torch.float32, device=self.device))
        matrix = torch.stack([raw[name] for name in self.component_names], dim=1)
        self.component_mean = matrix.mean(0)
        self.component_std = matrix.std(0).clamp_min(1e-4)
        normal_scores = self._combine(raw).detach()
        self.score_gate = torch.quantile(normal_scores, 0.99).item()
        self.entropy_gate = torch.quantile(raw["entropy"], 0.95).item()
        self.distance_gate = torch.quantile(raw["distance"], 0.99).item()
        self.margin_gate = torch.quantile(raw["margin"], 0.95).item()
        self.fit_seconds = time.perf_counter() - started
        return self

    def clone(self) -> "PlasticPrototypeMemory":
        return copy.deepcopy(self)

    def _responses(self, state: torch.Tensor):
        squared = torch.cdist(state, self.centers).square()
        logits = -squared / (2.0 * self.temperature * self.radii.square().unsqueeze(0).clamp_min(1e-8))
        response = torch.softmax(logits, dim=1)
        return response, squared.sqrt()

    def _raw_components(self, patches: torch.Tensor, use_attractor: bool = True):
        initial = patches
        response, _ = self._responses(initial)
        sorted_response = response.topk(k=min(2, response.shape[1]), dim=1).values
        if sorted_response.shape[1] == 1:
            margin = sorted_response[:, 0]
        else:
            margin = sorted_response[:, 0] - sorted_response[:, 1]
        entropy = -(response * response.clamp_min(1e-12).log()).sum(1)
        entropy = entropy / max(float(np.log(response.shape[1])), 1.0)

        state = initial
        previous = initial
        steps = self.attractor_steps if use_attractor else 0
        for _ in range(steps):
            previous = state
            response_t, _ = self._responses(state)
            target = response_t @ self.centers
            state = (1.0 - self.attractor_lr) * state + self.attractor_lr * target
        _, terminal_distances = self._responses(state)
        distance = (terminal_distances / self.radii.unsqueeze(0)).amin(1)
        residual = torch.linalg.vector_norm(state - initial, dim=1)
        stability = torch.linalg.vector_norm(state - previous, dim=1) if steps else torch.zeros_like(distance)
        components = {
            "distance": distance,
            "entropy": entropy,
            "margin": 1.0 - margin,
            "residual": residual,
            "stability": stability,
        }
        return components, response

    def _combine(self, raw: Mapping[str, torch.Tensor], enabled=None) -> torch.Tensor:
        enabled = set(enabled or self.component_names)
        score = torch.zeros_like(raw["distance"])
        for index, name in enumerate(self.component_names):
            if name in enabled:
                normalized = (raw[name] - self.component_mean[index]) / self.component_std[index]
                score = score + float(self.score_weights.get(name, 0.0)) * normalized
        return score

    @torch.inference_mode()
    def score(
        self,
        feature_map: np.ndarray,
        enabled=None,
        use_attractor: bool = True,
        use_group: bool = True,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        h, w, d = feature_map.shape
        patches = torch.as_tensor(feature_map.reshape(-1, d), dtype=torch.float32, device=self.device)
        raw, _ = self._raw_components(patches, use_attractor=use_attractor)
        score = self._combine(raw, enabled=enabled).reshape(h, w)
        if use_group and self.group_beta > 0:
            score = self._group_score(score, patches.reshape(h, w, d))
        return score.cpu().numpy(), {name: value.reshape(h, w).cpu().numpy() for name, value in raw.items()}

    def _group_score(self, score: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h, w, d = features.shape
        score_patches = F.unfold(score[None, None], kernel_size=3, padding=1).reshape(1, 9, h, w)
        feature_patches = F.unfold(
            features.permute(2, 0, 1)[None], kernel_size=3, padding=1
        ).reshape(1, d, 9, h, w)
        center = features.permute(2, 0, 1)[None, :, None]
        feature_dist = (feature_patches - center).square().sum(1)
        spatial = torch.tensor(
            [np.exp(-((i // 3 - 1) ** 2 + (i % 3 - 1) ** 2) / 2.0) for i in range(9)],
            dtype=features.dtype, device=features.device,
        ).reshape(1, 9, 1, 1)
        weights = spatial * torch.exp(-feature_dist / (2 * self.group_sigma_feature ** 2))
        neighbour = (weights * score_patches).sum(1) / weights.sum(1).clamp_min(1e-8)
        return (1.0 - self.group_beta) * score + self.group_beta * neighbour[0]

    @torch.inference_mode()
    def update(self, patches: np.ndarray, policy: str = "gated") -> int:
        x = torch.as_tensor(patches, dtype=torch.float32, device=self.device)
        raw, response = self._raw_components(x)
        winner = response.argmax(1)
        if policy == "fixed":
            return 0
        if policy == "ungated":
            accepted = torch.ones(len(x), dtype=torch.bool, device=self.device)
        elif policy == "gated":
            score = self._combine(raw)
            accepted = (
                (score < self.score_gate) &
                (raw["entropy"] < self.entropy_gate) &
                (raw["margin"] < self.margin_gate) &
                (raw["distance"] < self.distance_gate)
            )
        else:
            raise ValueError("policy must be fixed, ungated, or gated")
        candidates = accepted.nonzero(as_tuple=False).flatten()
        limit = max(1, int(len(x) * self.max_update_ratio))
        if len(candidates) > limit:
            confidence = self._combine(raw)[candidates]
            candidates = candidates[confidence.argsort()[:limit]]
        for index in candidates.tolist():
            k = int(winner[index])
            center = (1 - self.plasticity_lr) * self.centers[k] + self.plasticity_lr * x[index]
            self.centers[k] = center / center.norm().clamp_min(1e-8)
            distance = torch.linalg.vector_norm(x[index] - self.centers[k])
            self.radii[k] = (1 - self.plasticity_lr) * self.radii[k] + self.plasticity_lr * distance
        return len(candidates)

    @property
    def memory_bytes(self) -> int:
        return (
            self.centers.numel() * self.centers.element_size() +
            self.radii.numel() * self.radii.element_size()
        )


