"""Multi-task restorer: shared encoder, soft task routing and per-task heads."""

import torch
import torch.nn.functional as F
from torch import nn

from models.encoder import SharedEncoder
from models.heads.deblur import DeblurHead
from models.heads.dehaze import DehazeHead
from models.heads.denoise import DenoiseHead
from models.heads.derain import DerainHead
from models.heads.faceparse import FaceParseHead
from models.heads.facerestore import FaceRestoreHead
from models.heads.lowlight import LowLightHead
from models.heads.superres import SuperResHead
from models.heads.vintage import VintageHead
from training.losses import MULTITASK_HEAD_NAMES


class TaskClassifier(nn.Module):
    """Predict task logits from pooled encoder features."""

    def __init__(self, in_ch=64, num_tasks=11):
        super().__init__()
        # Adaptive pooling replaced with dynamic spatial mean for ONNX export compatibility
        self.fc = nn.Linear(in_ch, num_tasks)

    def forward(self, x):
        """Return task logits of shape [B, num_tasks]."""
        x = x.mean(dim=(-2, -1))
        logits = self.fc(x)
        return logits


class MultiTaskRestorer(nn.Module):
    """Shared-encoder restorer that blends per-task heads by predicted task weights."""

    head_param_prefixes: tuple[str, ...] = ("classifier.", "heads.")
    """Task classifier and the per-task restoration heads."""

    def __init__(self, num_tasks=11, **kwargs):
        super().__init__()

        self.encoder = SharedEncoder()
        self.classifier = TaskClassifier(in_ch=64, num_tasks=num_tasks)

        self.heads = nn.ModuleList(
            [
                DenoiseHead(),  # denoise
                DeblurHead(),  # deblur
                DerainHead(),  # derain
                DehazeHead(),  # dehaze_indoor
                DehazeHead(),  # dehaze_outdoor
                LowLightHead(),  # lowlight
                LowLightHead(),  # exposure
                SuperResHead(),  # superres
                VintageHead(),  # vintage
                FaceRestoreHead(),  # face_restorer
                FaceParseHead(),  # face_parser
            ]
        )

        self.task_names = list(MULTITASK_HEAD_NAMES)

    def forward(self, x, task=None):
        """Return (restored image, task logits); ``task`` forces a single head."""
        feat = self.encoder(x)

        # Soft Routing (Mixture of Experts)
        logits = self.classifier(feat)  # [B, num_tasks]
        weights = F.softmax(logits, dim=1)

        # If a specific task is requested (inference override), we can still support it
        # But by default, we use the learned weights
        if task is not None and task in self.task_names:
            idx = self.task_names.index(task)
            return self.heads[idx](feat), logits

        outputs = []
        for i, head in enumerate(self.heads):
            out = head(feat)
            outputs.append(out * weights[:, i].view(-1, 1, 1, 1))

        return torch.stack(outputs).sum(dim=0).contiguous(), logits
