import torch

def batched_dot_product(a, b):
    return torch.einsum('bn,bn->b', a, b)

def batched_max(a):
    return None

def select(a, index):
    return a[index]