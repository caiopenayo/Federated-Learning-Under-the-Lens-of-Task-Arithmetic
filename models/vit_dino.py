import torch
import torch.nn as nn
import timm


def build_dino_vit(
    num_classes=100,
    img_size=160,
    device=None,
    train_mode=True,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = timm.create_model(
        "vit_small_patch16_224.dino",
        pretrained=True,
        num_classes=0,     
        img_size=img_size, 
    )

    nn.init.normal_(model[1].weight, std=0.01)
    nn.init.zeros_(model[1].bias)

    model = nn.Sequential(
        backbone,
        nn.Linear(backbone.num_features, num_classes)
    ).to(device)

    if train_mode:
        model.train()

    return model
