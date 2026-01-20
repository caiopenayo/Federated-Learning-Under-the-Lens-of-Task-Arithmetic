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
    """
    Returns:
      client_indices: list[K] of lists of indices (relative to train_subset),
                      disjoint (no overlap) and without duplicates.
      client_classes: list[K] of lists with the classes assigned to each client.

    Properties:
      - Each client receives examples ONLY from Nc classes (client_classes[k]).
      - Examples are sampled WITHOUT replacement (disjoint).
      - Approximately equal sizes: distributes n//K to all and splits the remainder (n%K)
        among the first clients.
    """
    rng = np.random.default_rng(seed)

    n = len(train_subset)
    base_size = n // K
    remainder = n % K
    client_sizes = [base_size + (1 if k < remainder else 0) for k in range(K)]

    # 1) get labels from the subset (indices relative to train_subset)
    base_ds = train_subset.dataset
    subset_indices = np.array(train_subset.indices)  # indices in the base dataset
    labels = np.array(base_ds.targets)[subset_indices]  # labels aligned with the subset (len == n)

    # 2) group subset positions by class (lists will be "pools" consumed via pop)
    class_to_pool = defaultdict(list)
    for subset_pos, y in enumerate(labels):
        class_to_pool[int(y)].append(subset_pos)

    # shuffle each class pool
    for c in class_to_pool:
        rng.shuffle(class_to_pool[c])

    # 3) choose Nc classes per client, trying to avoid "oversubscription"
    # Strategy: always choose classes with the most remaining examples (greedy),
    # with random tie-breaking.
    client_classes = []
    for k in range(K):
        # candidate classes with at least 1 remaining example
        candidates = [c for c in range(num_classes) if len(class_to_pool.get(c, [])) > 0]
        if len(candidates) == 0:
            client_classes.append([])
            continue

        # sort by pool size (descending), with random noise for tie-breaking
        rng.shuffle(candidates)
        candidates.sort(key=lambda c: len(class_to_pool[c]), reverse=True)

        chosen = candidates[:Nc] if len(candidates) >= Nc else candidates
        client_classes.append(chosen)

    # 4) allocate examples to each client by consuming the pools (WITHOUT replacement)
    client_indices = [[] for _ in range(K)]

    for k in range(K):
        chosen = client_classes[k]
        target = client_sizes[k]
        if len(chosen) == 0 or target == 0:
            continue

        # round-robin over chosen classes to maintain mixing (when Nc > 1)
        ptr = 0
        while len(client_indices[k]) < target:
            c = chosen[ptr % len(chosen)]
            ptr += 1

            pool = class_to_pool.get(c, [])
            if len(pool) == 0:
                # this class is exhausted; check if any chosen class still has examples
                if all(len(class_to_pool.get(cc, [])) == 0 for cc in chosen):
                    break  # cannot complete while keeping the "only Nc classes" requirement
                continue

            # consume 1 example (disjoint!)
            client_indices[k].append(pool.pop())

    return client_indices, client_classes


def make_client_loaders(train_subset, client_indices, batch_size=64, num_workers=2):
    loaders = []
    for idxs in client_indices:
        if len(idxs) == 0:
            loaders.append(None)   # mark empty client
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
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val,   batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader

