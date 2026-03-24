import torch
import torch.nn as nn

class YOLOv8Mock(nn.Module):
    """
    Mock of YOLOv8n architecture.
    Takes 640x640 input and outputs [batch, 84, 8400] tensor.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8400)) # Flattening spatial dimension to match 8400 anchors
        )
        self.head = nn.Conv2d(32, 84, 1)

    def forward(self, x):
        feat = self.net(x)
        # feat is [b, 32, 1, 8400] -> [b, 84, 1, 8400] -> squeeze 2nd dim -> [b, 84, 8400]
        out = self.head(feat)
        return out.squeeze(2)

class RetinaFaceMock(nn.Module):
    """ 
    Mocks object detection backbones (MobileNet or ResNet) for face bounds. 
    Outputs: [B, 4] Bboxes, [B, 1] Confidence, [B, 10] Landmarks.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.bbox_head = nn.Linear(16, 4) # x1, y1, x2, y2
        self.conf_head = nn.Linear(16, 1) # confidence
        self.landmark_head = nn.Linear(16, 10) # 5 points * 2

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.bbox_head(feat), self.conf_head(feat), self.landmark_head(feat)
