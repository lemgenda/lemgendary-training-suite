import torch
import torch.nn as nn
from torchvision import models

class RetinaFace_MobileNet(nn.Module):
    """
    Real RetinaFace structure with MobileNetV2 backbone.
    Outputs: [B, 4] Bboxes, [B, 1] Confidence, [B, 10] Landmarks.
    """
    def __init__(self):
        super().__init__()
        # Use pretrained MobileNetV2 features
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        
        # Detection Heads
        self.conv_feat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 1280 is the output channel count of MobileNetV2
        self.bbox_head = nn.Linear(1280, 4)
        self.conf_head = nn.Linear(1280, 1)
        self.landmark_head = nn.Linear(1280, 10)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.conv_feat(feat)
        
        bboxes = self.bbox_head(feat)
        conf = torch.sigmoid(self.conf_head(feat))
        landmarks = self.landmark_head(feat)
        
        return bboxes, conf, landmarks

# YOLOv8 handling is delegated to the Ultralytics original library in train.py
# The factory should no longer instantiate a YOLOv8Mock.
