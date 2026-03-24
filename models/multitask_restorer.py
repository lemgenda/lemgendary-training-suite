import torch
import torch.nn as nn
from models.encoder import SharedEncoder
from models.heads.denoise import DenoiseHead
from models.heads.deblur import DeblurHead
from models.heads.derain import DerainHead
from models.heads.dehaze import DehazeHead
from models.heads.lowlight import LowLightHead
from models.heads.superres import SuperResHead

import torch.nn.functional as F

class TaskClassifier(nn.Module):
    def __init__(self, in_ch=64, num_tasks=6):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_tasks)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        return F.softmax(logits, dim=1)

class MultiTaskRestorer(nn.Module):
    def __init__(self, num_tasks=6):
        super().__init__()

        self.encoder = SharedEncoder()
        self.classifier = TaskClassifier(in_ch=64, num_tasks=num_tasks)

        self.heads = nn.ModuleList([
            DenoiseHead(),
            DeblurHead(),
            DerainHead(),
            DehazeHead(),
            LowLightHead(),
            SuperResHead()
        ])
        
        self.task_names = ["denoise", "deblur", "derain", "dehaze", "lowlight", "superres"]

    def forward(self, x, task=None):
        feat = self.encoder(x)
        
        # Soft Routing (Mixture of Experts)
        weights = self.classifier(feat) # [B, num_tasks]
        
        # If a specific task is requested (inference override), we can still support it
        # But by default, we use the learned weights
        if task is not None and task in self.task_names:
            idx = self.task_names.index(task)
            return self.heads[idx](feat), weights
            
        outputs = []
        for i, head in enumerate(self.heads):
            out = head(feat)
            outputs.append(out * weights[:, i].view(-1, 1, 1, 1))

        return sum(outputs), weights
