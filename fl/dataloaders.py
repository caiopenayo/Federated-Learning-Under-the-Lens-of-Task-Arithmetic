import torch
from torch.utils.data import DataLoader
from data.partition import iid_shard, noniid_shard_by_nc_disjoint, make_client_loaders
from data.datasets import get_cifar100, get_cifar100_transforms



def build_federated_dataloaders(
    train_transform,
    test_transform,
    K=100,
    sharding="iid",          # "iid" ou "non_iid"
    Nc=5,                    # usado só se sharding="non_iid"
    val_ratio=0.1,
    batch_size=64,
    num_workers=2,
    seed=42,
    root="./data"
):
    # 1) split train/val/test
    train, val, test = get_cifar100(
        train_transform=train_transform,
        test_transform=test_transform,
        val_ratio=val_ratio,
        root=root,
        seed=seed
    )

    # 2) shard do treino em K clientes
    if sharding == "iid":
        client_indices = iid_shard(train, K=K, seed=seed)
    elif sharding == "non_iid":
        client_indices, _ = noniid_shard_by_nc_disjoint(train, K=K, Nc=Nc, seed=seed)
    else:
        raise ValueError("sharding deve ser 'iid' ou 'non_iid'")

    # 3) dataloaders por cliente
    client_loaders = make_client_loaders(
        train_subset=train,
        client_indices=client_indices,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 4) val/test loaders (não federados)
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return client_loaders, val_loader, test_loader, train, val, test, client_indices
