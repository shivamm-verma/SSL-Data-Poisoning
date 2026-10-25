"""
Same frozen ImageNet-pretrained embedding extraction as the original
Phase 5, generalized to accept grayscale datasets (MNIST): a single
Grayscale->RGB conversion step is inserted before the standard resize +
ImageNet normalization, since the pretrained encoder expects 3 channels.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_embedding_transform(in_channels=3):
    steps = []
    if in_channels == 1:
        steps.append(T.Grayscale(num_output_channels=3))
    steps += [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(steps)


def build_pretrained_encoder(device):
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    encoder = tvm.resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


@torch.no_grad()
def extract_embeddings(dataset, encoder, device, batch_size=256, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=torch.cuda.is_available())
    all_emb, all_labels, all_idx, all_poisoned = [], [], [], []
    for images, labels, idx, is_poisoned in loader:
        images = images.to(device)
        emb = encoder(images)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())
        all_idx.append(idx.numpy())
        all_poisoned.append(np.asarray(is_poisoned))
    return (
        np.concatenate(all_emb, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_idx, axis=0),
        np.concatenate(all_poisoned, axis=0),
    )
