from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


def _normalize_level_indices(levels: Sequence[int], num_levels: int) -> Tuple[int, ...]:
    normalized = []
    for level in levels:
        idx = level + num_levels if level < 0 else level
        if 0 <= idx < num_levels:
            normalized.append(idx)
    return tuple(dict.fromkeys(normalized))


@MODELS.register_module()
class GeCoRegularizer(nn.Module):
    def __init__(
        self,
        enable: bool = True,
        alpha: float = 0.1,
        beta: float = 0.2,
        lambda_geo: float = 0.1,
        num_neighbors: int = 8,
        tangent_dim: int = 4,
        curvature_threshold: Optional[float] = None,
        curvature_quantile: float = 0.5,
        max_tokens: int = 1024,
        eps: float = 1e-6,
        detach_clean: bool = True,
        perturb_levels: Sequence[int] = (-1,),
        warmup_iters: int = 1000,
        prototype_momentum: float = 0.99,
        use_prototype_bank: bool = False,
        num_classes: int = 19,
        prototypes_per_class: int = 4,
    ) -> None:
        super().__init__()
        self.enable = enable
        self.alpha = alpha
        self.beta = beta
        self.lambda_geo = lambda_geo
        self.num_neighbors = num_neighbors
        self.tangent_dim = tangent_dim
        self.curvature_threshold = curvature_threshold
        self.curvature_quantile = curvature_quantile
        self.max_tokens = max_tokens
        self.eps = eps
        self.detach_clean = detach_clean
        self.perturb_levels = tuple(perturb_levels)
        self.warmup_iters = warmup_iters
        self.prototype_momentum = prototype_momentum
        self.use_prototype_bank = use_prototype_bank
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class

        self.register_buffer(
            "prototype_bank",
            torch.zeros(num_classes, prototypes_per_class, 1),
            persistent=False,
        )

    def forward(
        self,
        features,
        rein_deltas=None,
        labels=None,
        iteration: Optional[int] = None,
    ):
        if not self.enable:
            return features, {}

        feature_list, rebuild = self._unwrap_features(features)
        if rein_deltas is not None:
            delta_list, _ = self._unwrap_features(rein_deltas)
        else:
            delta_list = None
        level_indices = _normalize_level_indices(self.perturb_levels, len(feature_list))

        perturbed = []
        debug = {}
        for level, feature in enumerate(feature_list):
            if level not in level_indices or not isinstance(feature, Tensor):
                perturbed.append(feature)
                continue
            delta = None
            if delta_list is not None and level < len(delta_list):
                delta = delta_list[level]
            new_feature, level_debug = self._perturb_tensor(feature, delta)
            perturbed.append(new_feature)
            for key, value in level_debug.items():
                debug[f"level{level}_{key}"] = value

        return rebuild(perturbed), debug

    def geo_weight(self, iteration: Optional[int] = None) -> float:
        if self.warmup_iters <= 0 or iteration is None:
            return self.lambda_geo
        scale = min(float(iteration + 1) / float(self.warmup_iters), 1.0)
        return self.lambda_geo * scale

    def _unwrap_features(self, features):
        if isinstance(features, tuple) and len(features) == 2 and isinstance(features[0], (list, tuple)):
            first_type = type(features[0])

            def rebuild(items):
                return (first_type(items), features[1])

            return list(features[0]), rebuild

        if isinstance(features, list):
            return list(features), lambda items: list(items)
        if isinstance(features, tuple):
            return list(features), lambda items: tuple(items)
        return [features], lambda items: items[0]

    def _perturb_tensor(self, feature: Tensor, rein_delta: Optional[Tensor] = None):
        if feature.ndim == 4:
            return self._perturb_bchw(feature, rein_delta)
        if feature.ndim == 3:
            return self._perturb_bnc(feature, rein_delta)
        return feature, {}

    def _perturb_bchw(self, feature: Tensor, rein_delta: Optional[Tensor] = None):
        b, c, h, w = feature.shape
        pooled_feature = self._pool_feature(feature)
        delta = self._pool_feature(rein_delta) if rein_delta is not None else None
        ph, pw = pooled_feature.shape[-2:]
        tokens = pooled_feature.flatten(2).transpose(1, 2)
        delta_tokens = None if delta is None else delta.flatten(2).transpose(1, 2)

        perturb_tokens, debug = self._build_perturbation(tokens, delta_tokens)
        perturb = perturb_tokens.transpose(1, 2).reshape(b, c, ph, pw)
        if (ph, pw) != (h, w):
            perturb = F.interpolate(
                perturb,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        return feature + perturb.to(dtype=feature.dtype), debug

    def _perturb_bnc(self, feature: Tensor, rein_delta: Optional[Tensor] = None):
        b, n, c = feature.shape
        if n > self.max_tokens:
            indices = torch.linspace(0, n - 1, self.max_tokens, device=feature.device)
            indices = indices.round().long().unique()
            tokens = feature.index_select(1, indices)
            delta_tokens = None if rein_delta is None else rein_delta.index_select(1, indices)
        else:
            tokens = feature
            delta_tokens = rein_delta

        perturb_tokens, debug = self._build_perturbation(tokens, delta_tokens)
        if perturb_tokens.shape[1] != n:
            perturb_tokens = F.interpolate(
                perturb_tokens.transpose(1, 2),
                size=n,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return feature + perturb_tokens.to(dtype=feature.dtype), debug

    def _pool_feature(self, feature: Optional[Tensor]) -> Optional[Tensor]:
        if feature is None:
            return None
        if feature.ndim != 4:
            return feature
        _, _, h, w = feature.shape
        if h * w <= self.max_tokens:
            return feature
        aspect = h / max(w, 1)
        out_h = max(1, int((self.max_tokens * aspect) ** 0.5))
        out_w = max(1, self.max_tokens // out_h)
        while out_h * out_w > self.max_tokens:
            out_w = max(1, out_w - 1)
        return F.adaptive_avg_pool2d(feature, (out_h, out_w))

    def _build_perturbation(self, tokens: Tensor, rein_delta: Optional[Tensor]):
        b, n, c = tokens.shape
        if n <= 1 or c <= 0:
            return torch.zeros_like(tokens), {}

        k = min(self.num_neighbors, n - 1)
        d = min(self.tangent_dim, k)
        with torch.no_grad():
            geom_tokens = F.normalize(tokens.detach().float(), dim=-1, eps=self.eps)
            dist = torch.cdist(geom_tokens, geom_tokens)
            eye = torch.eye(n, dtype=torch.bool, device=tokens.device).unsqueeze(0)
            dist = dist.masked_fill(eye, float("inf"))
            knn_idx = dist.topk(k, dim=-1, largest=False).indices

            batch_idx = torch.arange(b, device=tokens.device).view(b, 1, 1)
            neighbors = geom_tokens[batch_idx, knn_idx]
            centers = geom_tokens.unsqueeze(2)
            offsets = neighbors - centers

            gram = torch.matmul(offsets, offsets.transpose(-1, -2))
            gram = gram + self.eps * torch.eye(k, device=tokens.device).view(1, 1, k, k)
            eigvals, eigvecs = torch.linalg.eigh(gram)
            top_vals = eigvals[..., -d:].clamp_min(self.eps)
            top_vecs = eigvecs[..., -d:]

            projected_offsets = torch.matmul(
                top_vecs,
                torch.matmul(top_vecs.transpose(-1, -2), offsets),
            )
            residual = offsets - projected_offsets
            curvature = residual.pow(2).sum(-1) / (offsets.pow(2).sum(-1) + self.eps)
            curvature = torch.nan_to_num(
                curvature.mean(-1),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            curvature = curvature.clamp(0.0, 1.0)

            inv_sqrt = top_vals.rsqrt()
            basis = torch.einsum("bnkd,bnkc,bnd->bndc", top_vecs, offsets, inv_sqrt)

        random_dir = torch.randn_like(tokens)
        basis = basis.to(dtype=random_dir.dtype)
        tangent_coeff = torch.einsum("bnc,bndc->bnd", random_dir, basis)
        tangent_component = torch.einsum("bnd,bndc->bnc", tangent_coeff, basis)
        normal_component = random_dir - tangent_component

        direction_high = F.normalize(tangent_component, dim=-1, eps=self.eps)
        direction_low = F.normalize(
            tangent_component + self.beta * normal_component,
            dim=-1,
            eps=self.eps,
        )

        if self.curvature_threshold is None:
            threshold = torch.quantile(
                curvature.detach().float(),
                q=self.curvature_quantile,
                dim=1,
                keepdim=True,
            )
        else:
            threshold = curvature.new_full((b, 1), float(self.curvature_threshold))

        high_mask = (curvature >= threshold).unsqueeze(-1)
        direction = torch.where(high_mask, direction_high, direction_low)
        direction = F.normalize(direction, dim=-1, eps=self.eps)

        epsilon = self.alpha / (1.0 + curvature)
        epsilon = epsilon.clamp(min=0.0, max=self.alpha).unsqueeze(-1)
        epsilon = epsilon.to(dtype=tokens.dtype)

        if rein_delta is not None and rein_delta.shape == tokens.shape:
            perturbation = epsilon * direction * rein_delta
        else:
            scale = tokens.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
            perturbation = epsilon * direction * scale

        perturbation = torch.nan_to_num(perturbation, nan=0.0, posinf=0.0, neginf=0.0)
        debug = {
            "curvature_mean": curvature.mean().detach(),
            "curvature_max": curvature.max().detach(),
            "epsilon_mean": epsilon.detach().float().mean(),
        }
        return perturbation, debug
