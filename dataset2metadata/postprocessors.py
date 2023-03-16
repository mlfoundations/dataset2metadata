import torch
from typing import Dict, Any

def _cpu_helper(entry: Any, to_cpu: bool):
    if to_cpu:
        return entry.cpu()
    return entry

def batched_dot_product(cache: Dict, k: str, to_cpu: bool=True):
    return _cpu_helper(torch.einsum('bn,bn->b', cache[k][0], cache[k][1]), to_cpu)

def select(cache: Dict, k: str, index: int, to_cpu: bool=True):
    return _cpu_helper(cache[k][index], to_cpu)

def identity(cache: Dict, k: str, to_cpu: bool=True):
    return _cpu_helper(cache[k], to_cpu)

def transpose_list(cache: Dict, k: str):
    return list(map(list, zip(*cache[k])))