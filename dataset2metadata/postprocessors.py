import torch
from typing import Dict


def batched_dot_product(d: Dict, a: str, b: str):
    return torch.einsum('bn,bn->b', d[a], d[b])

def batched_max(d: Dict, a: str):
    return None

def select(d: Dict, a: str, index: int):
    return a[index]

def identity(d: Dict, a: str):
    return d[a]