import numpy as np
from collections import defaultdict
from torch.utils.data import Subset, DataLoader

def iid_shard(train_subset, K, seed=42):
    n = len(train_subset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    shards = np.array_split(indices, K)
    return [shard.tolist() for shard in shards]

def noniid_shard_by_nc_disjoint(train_subset, K, Nc, seed=42, num_classes=100):
    rng = np.random.default_rng(seed)

    n = len(train_subset)
    base_size = n // K
    remainder = n % K
    client_sizes = [base_size + (1 if k < remainder else 0) for k in range(K)]

    base_ds = train_subset.dataset
    subset_indices = np.array(train_subset.indices)  
    labels = np.array(base_ds.targets)[subset_indices]  

    class_to_pool = defaultdict(list)
    for subset_pos, y in enumerate(labels):
        class_to_pool[int(y)].append(subset_pos)

    for c in class_to_pool:
        rng.shuffle(class_to_pool[c])

   
    client_indices = [[] for _ in range(K)]
    client_classes = [[] for _ in range(K)]

    for k in range(K):
        target = client_sizes[k]
        if target == 0:
            continue

        candidates = [c for c in range(num_classes) if len(class_to_pool.get(c, [])) > 0]
        if len(candidates) == 0:
            continue

        rng.shuffle(candidates)
        candidates.sort(key=lambda c: len(class_to_pool[c]), reverse=True)

        chosen = candidates[:Nc] if len(candidates) >= Nc else candidates
        client_classes[k] = chosen

        ptr = 0
        while len(client_indices[k]) < target:
            c = chosen[ptr % len(chosen)]
            ptr += 1

            pool = class_to_pool.get(c, [])
            if len(pool) == 0:
                if all(len(class_to_pool.get(cc, [])) == 0 for cc in chosen):
                    break  
                continue

            client_indices[k].append(pool.pop())

    return client_indices, client_classes


def make_client_loaders(train_subset, client_indices, batch_size=64, num_workers=2):
    loaders = []
    for idxs in client_indices:
        if len(idxs) == 0:
            loaders.append(None)   
            continue

        ds_k = Subset(train_subset, idxs)
        dl_k = DataLoader(
            ds_k,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders.append(dl_k)
    return loaders

def make_dataset_loaders(train, val, test):
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val,   batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, test_loader
