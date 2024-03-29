import torch
from typing import Dict, Any


def _cpu_helper(entry: Any, to_cpu: bool):
    if to_cpu:
        return entry.cpu()
    return entry


def batched_dot_product_index(
    cache: Dict, model: str, i: int, j: int, to_cpu: bool = True
):
    return _cpu_helper(
        torch.einsum("bn,bn->b", cache[model][i], cache[model][j]), to_cpu
    )


def batched_dot_product(cache: Dict, model: str, to_cpu: bool = True):
    return batched_dot_product_index(cache, model, 0, 1, to_cpu=to_cpu)


def select(cache: Dict, model: str, index: int, to_cpu: bool = True):
    return _cpu_helper(cache[model][index], to_cpu)


def identity(cache: Dict, model: str, to_cpu: bool = True):
    return _cpu_helper(cache[model], to_cpu)


def transpose_list(cache: Dict, model: str):
    return list(map(list, zip(*cache[model])))
