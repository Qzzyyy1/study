from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.WGDT import Model as BaseModel


class OVAHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        ova_loss_weight: float = 1.0,
        open_entropy_weight: float = 0.1,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ova_loss_weight = ova_loss_weight
        self.open_entropy_weight = open_entropy_weight
        self.threshold = threshold
        self.closed_classifier = nn.Linear(in_features, num_classes)
        self.ova_classifier = nn.Linear(in_features, num_classes)

    def build_ova_targets(self, labels: torch.Tensor) -> torch.Tensor:
        target = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        target.scatter_(1, labels.view(-1, 1), 1.0)
        return target

    def compute_source_loss(
        self,
        closed_logits: torch.Tensor,
        ova_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        loss_cls = F.cross_entropy(closed_logits, labels)
        ova_targets = self.build_ova_targets(labels)
        loss_ova = F.binary_cross_entropy_with_logits(ova_logits, ova_targets)
        total = loss_cls + self.ova_loss_weight * loss_ova
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_ova": loss_ova,
        }

    def compute_open_entropy(self, ova_logits: torch.Tensor) -> torch.Tensor:
        ova_prob = torch.sigmoid(ova_logits)
        entropy = -(
            ova_prob * torch.log(ova_prob + 1e-12)
            + (1 - ova_prob) * torch.log(1 - ova_prob + 1e-12)
        )
        return entropy.mean()

    def forward(self, features: torch.Tensor, y: Optional[torch.Tensor] = None) -> dict:
        closed_logits = self.closed_classifier(features)
        closed_prob = F.softmax(closed_logits, dim=1)
        prediction = closed_prob.argmax(dim=1)

        ova_logits = self.ova_classifier(features)
        ova_prob = torch.sigmoid(ova_logits)
        known_score = torch.gather(ova_prob, 1, prediction.view(-1, 1)).view(-1)
        ova_confidence = ova_prob.max(dim=1)[0]

        out = {
            "closed_logits": closed_logits,
            "closed_prob": closed_prob,
            "ova_logits": ova_logits,
            "ova_prob": ova_prob,
            "prediction": prediction,
            "known_score": known_score,
            "ova_confidence": ova_confidence,
        }

        if y is not None:
            out.update(self.compute_source_loss(closed_logits, ova_logits, y))
        else:
            out["loss_open_entropy"] = self.compute_open_entropy(ova_logits)

        return out


class EMAOptimizerWrapper:
    def __init__(self, optimizer, model, momentum: float):
        self.optimizer = optimizer
        self.model = model
        self.momentum = momentum

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.model.update_teacher(self.momentum)


class Model(BaseModel):
    def __init__(
        self,
        *args,
        ova_loss_weight: float = 1.0,
        open_entropy_weight: float = 0.1,
        ova_threshold: float = 0.5,
        consistency_loss_weight: float = 0.1,
        consistency_threshold: float = 0.7,
        feature_noise_std: float = 0.05,
        density_quantile: float = 0.05,
        rescue_margin: float = 0.15,
        variance_floor: float = 1e-4,
        teacher_momentum: float = 0.999,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        feature_dim = self.classifier.in_features
        self.ova_loss_weight = ova_loss_weight
        self.open_entropy_weight = open_entropy_weight
        self.ova_threshold = ova_threshold
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_threshold = consistency_threshold
        self.feature_noise_std = feature_noise_std
        self.density_quantile = density_quantile
        self.rescue_margin = rescue_margin
        self.variance_floor = variance_floor
        self.teacher_momentum = teacher_momentum

        self.head = OVAHead(
            in_features=feature_dim,
            num_classes=self.known_num_classes,
            ova_loss_weight=ova_loss_weight,
            open_entropy_weight=open_entropy_weight,
            threshold=ova_threshold,
        )

        self.register_buffer("class_feature_mean", torch.zeros(self.known_num_classes, feature_dim))
        self.register_buffer("class_feature_var", torch.ones(self.known_num_classes, feature_dim))
        self.register_buffer("class_density_thresholds", torch.zeros(self.known_num_classes))
        self.register_buffer("density_ready", torch.zeros(1))

        self.teacher_feature_encoder = copy.deepcopy(self.feature_encoder)
        self.teacher_head = copy.deepcopy(self.head)
        self.freeze_teacher()

    def freeze_teacher(self):
        self.teacher_feature_encoder.eval()
        self.teacher_head.eval()
        for parameter in self.teacher_feature_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.teacher_head.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None):
        ema_momentum = self.teacher_momentum if momentum is None else float(momentum)

        for teacher_param, student_param in zip(
            self.teacher_feature_encoder.parameters(),
            self.feature_encoder.parameters(),
        ):
            teacher_param.data.mul_(ema_momentum).add_(student_param.data * (1.0 - ema_momentum))
        for teacher_param, student_param in zip(self.teacher_head.parameters(), self.head.parameters()):
            teacher_param.data.mul_(ema_momentum).add_(student_param.data * (1.0 - ema_momentum))

        for teacher_buffer, student_buffer in zip(
            self.teacher_feature_encoder.buffers(),
            self.feature_encoder.buffers(),
        ):
            teacher_buffer.data.copy_(student_buffer.data)
        for teacher_buffer, student_buffer in zip(self.teacher_head.buffers(), self.head.buffers()):
            teacher_buffer.data.copy_(student_buffer.data)

    @torch.no_grad()
    def teacher_forward(self, x: torch.Tensor) -> dict:
        self.teacher_feature_encoder.eval()
        self.teacher_head.eval()
        features = self.teacher_feature_encoder(x)["features"]
        head_out = self.teacher_head(features)
        return {
            "features": features,
            **head_out,
        }

    def forward(self, x, y=None):
        features = self.feature_encoder(x)["features"]
        head_out = self.head(features, y)
        return {
            "features": features,
            **head_out,
        }

    def compute_teacher_consistency_loss(self, teacher_out: dict, student_noisy_out: dict) -> torch.Tensor:
        teacher_known_score = teacher_out["known_score"].detach()
        mask = teacher_known_score >= float(self.consistency_threshold)
        if not bool(mask.any()):
            return teacher_known_score.new_zeros(())

        teacher_ova_prob = teacher_out["ova_prob"].detach()
        student_ova_prob = student_noisy_out["ova_prob"]
        per_sample_loss = F.mse_loss(student_ova_prob, teacher_ova_prob, reduction="none").mean(dim=1)
        weights = teacher_known_score[mask]
        weighted_loss = per_sample_loss[mask] * weights
        return weighted_loss.sum() / torch.clamp(weights.sum(), min=1e-6)

    @torch.no_grad()
    def calibrate_density_statistics(self, dataloader: DataLoader) -> None:
        self.eval()
        features_by_class = [[] for _ in range(self.known_num_classes)]

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = self(x)
            features = out["features"].detach()
            for class_index in range(self.known_num_classes):
                class_mask = y == class_index
                if class_mask.any():
                    features_by_class[class_index].append(features[class_mask].detach().cpu())

        means = []
        variances = []
        thresholds = []
        for class_index in range(self.known_num_classes):
            if not features_by_class[class_index]:
                raise RuntimeError(f"No source features collected for class {class_index}")
            class_features = torch.cat(features_by_class[class_index], dim=0).float()
            class_mean = class_features.mean(dim=0)
            class_var = class_features.var(dim=0, unbiased=False).clamp_min(self.variance_floor)
            normalized_distance = ((class_features - class_mean) ** 2 / class_var).mean(dim=1)
            density_score = torch.exp(-0.5 * normalized_distance)
            density_threshold = torch.quantile(density_score, self.density_quantile)
            means.append(class_mean)
            variances.append(class_var)
            thresholds.append(density_threshold)

        self.class_feature_mean.copy_(torch.stack(means).to(self.class_feature_mean.device))
        self.class_feature_var.copy_(torch.stack(variances).to(self.class_feature_var.device))
        self.class_density_thresholds.copy_(torch.stack(thresholds).to(self.class_density_thresholds.device))
        self.density_ready.fill_(1.0)
        self.train()

    def density_score_for_prediction(self, features: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        class_mean = self.class_feature_mean[prediction]
        class_var = self.class_feature_var[prediction].clamp_min(self.variance_floor)
        normalized_distance = ((features - class_mean) ** 2 / class_var).mean(dim=1)
        return torch.exp(-0.5 * normalized_distance)

    def density_threshold_for_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        return self.class_density_thresholds[prediction]

    def should_rescue(
        self,
        known_score: torch.Tensor,
        density_score: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.density_ready.item()):
            return torch.zeros_like(known_score, dtype=torch.bool)
        lower_bound = self.ova_threshold - self.rescue_margin
        borderline_mask = (known_score < self.ova_threshold) & (known_score >= lower_bound)
        density_threshold = self.density_threshold_for_prediction(prediction)
        density_mask = density_score >= density_threshold
        return borderline_mask & density_mask

    def pre_train_step(self, batch):
        x, y = batch
        out = self(x, y)
        information = {
            "loss_cls": out["loss_cls"],
            "loss_ova": out["loss_ova"] * self.ova_loss_weight,
        }
        return {
            "loss": out["loss"],
            "information": information,
        }

    def pre_train_optimizer(self):
        optimizer = torch.optim.SGD(
            [
                {"params": self.feature_encoder.parameters()},
                {"params": self.head.parameters()},
            ],
            lr=0.001,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return EMAOptimizerWrapper(optimizer, self, self.teacher_momentum)

    def train_step(self, batch):
        [source_x, source_y], [target_x, target_y] = batch

        source_out = self(source_x, source_y)
        target_out = self(target_x)

        noise = torch.randn_like(target_out["features"]) * float(self.feature_noise_std)
        noisy_target_out = self.head(target_out["features"] + noise)
        teacher_target_out = self.teacher_forward(target_x)
        loss_consistency = self.compute_teacher_consistency_loss(teacher_target_out, noisy_target_out)

        target_weight = target_out["ova_confidence"].detach()
        loss_disc = self.domain_adv(source_out["features"], target_out["features"], w_t=target_weight)
        loss_open_entropy = target_out["loss_open_entropy"]

        loss_dic = dict(
            loss_cls=source_out["loss_cls"],
            loss_ova=source_out["loss_ova"] * self.ova_loss_weight,
            loss_disc=loss_disc * self.domain_loss_weight,
            loss_open_entropy=loss_open_entropy * self.open_entropy_weight,
            loss_consistency=loss_consistency * self.consistency_loss_weight,
        )

        self.source_oa.update(source_out["prediction"], source_y)

        return {
            "loss": sum(value for value in loss_dic.values()),
            "information": loss_dic,
        }

    def train_optimizer(self):
        momentum = 0.9
        l2_decay = 5e-4
        optimizer = torch.optim.SGD(
            [
                {"params": self.feature_encoder.parameters()},
                {"params": self.head.parameters()},
            ],
            lr=self.args.lr_encoder,
            momentum=momentum,
            weight_decay=l2_decay,
        )
        optimizer_critic = torch.optim.SGD(
            self.disc_encoder.parameters(),
            lr=self.args.lr_domain,
            momentum=momentum,
            weight_decay=l2_decay,
        )
        optimizer = EMAOptimizerWrapper(optimizer, self, self.teacher_momentum)
        return optimizer, optimizer_critic

    def test_step(self, batch):
        x, y = batch
        out = self(x)
        prediction = out["prediction"].clone()
        reject_mask = out["known_score"] < self.ova_threshold
        density_score = self.density_score_for_prediction(out["features"], prediction)
        rescue_mask = self.should_rescue(out["known_score"], density_score, prediction)
        prediction[reject_mask & ~rescue_mask] = self.known_num_classes
        self.metric.update(prediction, y)

    def prediction_step(self, batch):
        x = batch
        out = self(x)
        prediction = out["prediction"].clone()
        reject_mask = out["known_score"] < self.ova_threshold
        density_score = self.density_score_for_prediction(out["features"], prediction)
        rescue_mask = self.should_rescue(out["known_score"], density_score, prediction)
        prediction[reject_mask & ~rescue_mask] = self.known_num_classes
        self.prediciton_all.append(prediction)


__all__ = ["EMAOptimizerWrapper", "Model", "OVAHead"]
