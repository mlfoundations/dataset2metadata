import hashlib
import os
import random
from typing import List, Dict, Set
import urllib
import warnings
from tqdm import tqdm

import numpy as np
import torch


def random_seed(seed: int = 0) -> None:
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

def download(url: str, root: str):
    # modified from oai _download clip function
    print(url)
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url).split('_')[-1]

    expected_sha256 = os.path.basename(url).split('_')[0]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target