import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
INPUT_SIZE = 224

def get_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.03),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model

def get_transform():
    return transforms.Compose([
        transforms.Raize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize()
        ])