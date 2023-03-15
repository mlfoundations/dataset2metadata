import random
from typing import List, Dict, Set

import numpy as np
import torch


def random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def topsort(graph: Dict[str, List[str]]) -> List[str]:
    # from: https://stackoverflow.com/questions/52432988/python-dict-key-order-based-on-values-recursive-solution
    result: List[str] = []
    seen: Set[str] = set()

    def recursive_helper(node: str) -> None:
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                recursive_helper(neighbor)
        if node not in result:
            result.append(node)

    for key in graph.keys():
        recursive_helper(key)

    return result
