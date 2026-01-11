import torch
import torch.nn as nn
import timm


def build_dino_vit(
    num_classes=100,
    img_size=160,
    device=None,
    train_mode=True,
    pretrained: bool = True,
):
    """
    Retorna um modelo DINO ViT-S/16 com uma head Linear para CIFAR-100.

    - Usa backbone timm: vit_small_patch16_224.dino
    - Permite img_size != 224 (timm faz interpolação de pos_embed em geral)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = timm.create_model(
        "vit_small_patch16_224.dino",
        pretrained=pretrained,
        num_classes=0,     # retorna features
        img_size=img_size, # <-- chave para acelerar (160)
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
