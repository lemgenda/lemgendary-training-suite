import torch
import torch.nn as nn
from torchvision import models

class RetinaFace_MobileNet(nn.Module):
    """
    Real RetinaFace structure with MobileNetV3-Small backbone.
    Outputs: [B, 4] Bboxes, [B, 1] Confidence, [B, 10] Landmarks.
    """
    def __init__(self, backbone="mobilenet_v3_small", **kwargs):
        super().__init__()
        if backbone == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            in_channels = 1280
        else:
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
            in_channels = 576
        
        # Detection Heads
        self.conv_feat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.bbox_head = nn.Linear(in_channels, 4)
        self.conf_head = nn.Linear(in_channels, 1)
        self.landmark_head = nn.Linear(in_channels, 10)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.conv_feat(feat)
        
        bboxes = self.bbox_head(feat)
        conf = torch.sigmoid(self.conf_head(feat))
        landmarks = self.landmark_head(feat)
        
        return bboxes, conf, landmarks

# YOLOv8 handling is delegated to the Ultralytics original library in train.py
# The factory should no longer instantiate a YOLOv8Mock.
