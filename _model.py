import torch
import torch.nn as nn
import timm


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class EfficientV2Base(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=True)

    def forward(self, x):
        x = torch.sigmoid(self.backbone(x))
        return x


class EfficientV2(nn.Module):
    def __init__(self, num_classes=518):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=True, num_classes=num_classes
        )
        nn.init.xavier_normal_(self.backbone.classifier.weight)

    def forward(self, x):
        x = torch.sigmoid(self.backbone(x))
        return x


class EfficientV2Backbone(nn.Module):
    def __init__(self, model=None, num_classes=5):
        super().__init__()
        if model == None:
            self.backbone = torch.load("./ckpt.pth")
        else:
            self.backbone = model
        self.backbone.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x
