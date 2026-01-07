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
    Retorna:
      client_indices: list[K] de listas de índices (relativos ao train_subset),
                      disjuntos (sem overlap) e sem duplicatas.
      client_classes: list[K] de listas com as classes atribuídas a cada cliente.

    Propriedades:
      - Cada cliente recebe exemplos APENAS de Nc classes (client_classes[k]).
      - Os exemplos são amostrados SEM reposição (disjunto).
      - Tamanhos ~iguais: distribui n//K para todos e reparte o resto (n%K) nos primeiros clientes.
    """
    rng = np.random.default_rng(seed)

    n = len(train_subset)
    base_size = n // K
    remainder = n % K
    client_sizes = [base_size + (1 if k < remainder else 0) for k in range(K)]

    # 1) obter labels do subset (índices relativos ao train_subset)
    base_ds = train_subset.dataset
    subset_indices = np.array(train_subset.indices)  # índices no dataset base
    labels = np.array(base_ds.targets)[subset_indices]  # labels alinhados ao subset (len == n)

    # 2) agrupar posições do subset por classe (listas serão "pools" que vamos consumir via pop)
    class_to_pool = defaultdict(list)
    for subset_pos, y in enumerate(labels):
        class_to_pool[int(y)].append(subset_pos)

    # embaralha cada pool de classe
    for c in class_to_pool:
        rng.shuffle(class_to_pool[c])

    # 3) escolher Nc classes por cliente, tentando evitar "oversubscription"
    # Estratégia: sempre escolher as classes com mais exemplos restantes (greedy) com desempate aleatório.
    client_classes = []
    for k in range(K):
        # classes candidatas com pelo menos 1 exemplo restante
        candidates = [c for c in range(num_classes) if len(class_to_pool.get(c, [])) > 0]
        if len(candidates) == 0:
            client_classes.append([])
            continue

        # ordena por tamanho do pool (desc), com ruído aleatório pra desempate
        rng.shuffle(candidates)
        candidates.sort(key=lambda c: len(class_to_pool[c]), reverse=True)

        chosen = candidates[:Nc] if len(candidates) >= Nc else candidates
        client_classes.append(chosen)

    # 4) alocar exemplos para cada cliente consumindo as pools (SEM reposição)
    client_indices = [[] for _ in range(K)]

    for k in range(K):
        chosen = client_classes[k]
        target = client_sizes[k]
        if len(chosen) == 0 or target == 0:
            continue

        # round-robin nas classes escolhidas para manter mistura (quando Nc>1)
        ptr = 0
        while len(client_indices[k]) < target:
            c = chosen[ptr % len(chosen)]
            ptr += 1

            pool = class_to_pool.get(c, [])
            if len(pool) == 0:
                # essa classe acabou; verifica se ainda existe alguma das classes escolhidas com exemplos
                if all(len(class_to_pool.get(cc, [])) == 0 for cc in chosen):
                    break  # não dá pra completar mantendo o requisito "somente Nc classes"
                continue

            # consome 1 exemplo (disjunto!)
            client_indices[k].append(pool.pop())

    return client_indices, client_classes


def make_client_loaders(train_subset, client_indices, batch_size=64, num_workers=2):
    loaders = []
    for idxs in client_indices:
        if len(idxs) == 0:
            loaders.append(None)   # marca cliente vazio
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