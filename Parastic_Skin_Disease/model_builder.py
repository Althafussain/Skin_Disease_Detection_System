import torch.nn as nn
from torchvision import models


def build_parasitic_model(num_classes, train_from_scratch=True):
    weights = None
    if not train_from_scratch:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1

    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
