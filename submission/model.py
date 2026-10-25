"""Submission Studio 2 - convnext_tiny (timm). get_model() hanya membangun arsitektur."""
import timm
from torchvision import transforms

ARCH = "convnext_tiny"
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Urutan kelas: 0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam


def get_model():
    return timm.create_model(ARCH, pretrained=False, num_classes=5)  # logits [B, 5]


def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
