import torch
import torch.nn as nn
import timm


def build_dino_vit(
    num_classes=100,
    img_size=160,
    device=None,
    train_mode=True,
):
    """
    Returns a DINO ViT-S/16 model with a Linear head for CIFAR-100.

    - Uses backbone timm: vit_small_patch16_224.dino

    - Allows img_size != 224 (timm generally performs pos_embed interpolation)

    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = timm.create_model(
        "vit_small_patch16_224.dino",
        pretrained=True,
        num_classes=0,     # return features
        img_size=img_size, # <-- key to speedup (160)
    )


    model = nn.Sequential(
        backbone,
        nn.Linear(backbone.num_features, num_classes)
    ).to(device)

    nn.init.normal_(model[1].weight, std=0.01)
    nn.init.zeros_(model[1].bias)

    if train_mode:
        model.train()

    return model
