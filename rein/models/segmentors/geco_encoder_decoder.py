from typing import Optional

import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from torch import Tensor

from ..utils import GeCoRegularizer  # noqa: F401
from .frozen_encoder_decoder import PEFTBackboneEncoderDecoder


@MODELS.register_module()
class GeCoPEFTBackboneEncoderDecoder(PEFTBackboneEncoderDecoder):
    def __init__(
        self,
        enable_geco: bool = False,
        geco_regularizer: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_geco = enable_geco
        geco_cfg = dict(type="GeCoRegularizer", enable=enable_geco)
        if geco_regularizer is not None:
            geco_cfg.update(geco_regularizer)
            geco_cfg["enable"] = enable_geco
        self.geco_regularizer = MODELS.build(geco_cfg)
        self.register_buffer("_geco_iter", torch.zeros((), dtype=torch.long), persistent=False)

    def loss(self, inputs: Tensor, batch_data_samples: list) -> dict:
        if not self.training or not self.enable_geco:
            return super().loss(inputs, batch_data_samples)

        features = self.extract_feat(inputs)
        losses = self._decode_head_forward_train(features, batch_data_samples)
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(features, batch_data_samples))

        iteration = int(self._geco_iter.item())
        if iteration < self.geco_regularizer.warmup_iters:
            losses["loss_geo"] = self._zero_like_features(features)
            self._geco_iter += 1
            return losses

        perturbed_features, debug_info = self.geco_regularizer(
            features,
            labels=self._stack_labels(batch_data_samples),
            iteration=iteration,
        )

        if self.geco_regularizer.detach_clean:
            with torch.no_grad():
                clean_logits = self._decode_logits(features, batch_data_samples)
        else:
            clean_logits = self._decode_logits(features, batch_data_samples)
        perturbed_logits = self._decode_logits(perturbed_features, batch_data_samples)
        loss_geo = self._geodesic_consistency(clean_logits, perturbed_logits)
        losses["loss_geo"] = loss_geo * self.geco_regularizer.geo_weight(iteration)

        for name, value in debug_info.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                losses[f"geco_{name}"] = value

        self._geco_iter += 1
        return losses

    def _decode_logits(self, features, batch_data_samples: list) -> Tensor:
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        return self.decode_head.predict(features, batch_img_metas, self.test_cfg)

    def _geodesic_consistency(self, clean_logits: Tensor, perturbed_logits: Tensor) -> Tensor:
        if perturbed_logits.shape[-2:] != clean_logits.shape[-2:]:
            perturbed_logits = F.interpolate(
                perturbed_logits,
                size=clean_logits.shape[-2:],
                mode="bilinear",
                align_corners=getattr(self.decode_head, "align_corners", False),
            )

        if self.geco_regularizer.detach_clean:
            clean_logits = clean_logits.detach()

        clean_logits = torch.nan_to_num(
            clean_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0
        ).clamp(-30.0, 30.0)
        perturbed_logits = torch.nan_to_num(
            perturbed_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0
        ).clamp(-30.0, 30.0)
        clean_prob = clean_logits.softmax(dim=1).clamp_min(self.geco_regularizer.eps)
        perturbed_prob = perturbed_logits.softmax(dim=1).clamp_min(
            self.geco_regularizer.eps
        )
        inner = (clean_prob.sqrt() * perturbed_prob.sqrt()).sum(dim=1)
        inner = inner.clamp(
            min=-1.0 + self.geco_regularizer.eps,
            max=1.0 - self.geco_regularizer.eps,
        )
        return torch.nan_to_num(torch.acos(inner).pow(2).mean(), nan=0.0, posinf=0.0)

    def _zero_like_features(self, features) -> Tensor:
        if isinstance(features, tuple) and features:
            return self._zero_like_features(features[0])
        if isinstance(features, list):
            return self._zero_like_features(features[0])
        return features.new_zeros(())

    def _stack_labels(self, batch_data_samples: list) -> Optional[Tensor]:
        labels = []
        for sample in batch_data_samples:
            if not hasattr(sample, "gt_sem_seg"):
                return None
            labels.append(sample.gt_sem_seg.data)
        if not labels:
            return None
        return torch.stack(labels, dim=0)
