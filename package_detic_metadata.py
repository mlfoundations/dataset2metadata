from PIL import Image, ImageDraw
import json
import numpy as np
import math
import webdataset as wds
from dataset2metadata.dataloaders import tarfile_to_samples_nothrow, json_decoder
from functools import partial
from io import BytesIO
import s3fs
from tqdm import tqdm
import pandas as pd

s3 = s3fs.S3FileSystem(anon=False)

# mini = float('inf')
# maxi = float('-inf')
# for e in tqdm(s3.ls("s3://dcnlp-hub/datacomp_rebuttal/")):
#     counts = [len(ee.split('/')[-1].split('.')[0]) == 7 for ee in s3.ls(e)]
#     num = [int(ee.split('/')[-1].split('.')[0]) for ee in s3.ls(e)]

#     if max(num) > maxi:
#         maxi = max(num)
#     if min(num) < mini:
#         mini = min(num)
#     # print(num)
#     assert all(counts)
#     # print(s3.ls(e))
# print(maxi)
# print(mini)
# print('great.')
# exit(0)

# input_shards = "pipe:aws s3 cp s3://dcnlp-hub/datacomp_rebuttal/shard_{0000..0255}/{0000000..0000050}.tar -"
batch_size = 2048
nworkers = 4


def construct_lvis_index(path="lvis2.json"):#_val_100.json"):
    meta = None
    index = {}
    with open(path, "r") as f:
        meta = json.load(f)#["categories"]

    lvis_cats = sorted(meta, key=lambda x: x["id"])
    index = [cat["name"] for cat in lvis_cats]

    return index

def normalized_center_to_grid_loc(center_x, center_y, n=5):
    return math.floor(center_x * n), math.floor(center_y * n)

def detic_decoder(key, value):
    if key.endswith("scores") or key.endswith("classes") or key.endswith("boxes"):
        return str(np.load(BytesIO(value)).tolist())

    return None

all_stuff = s3.ls("s3://dcnlp-hub/datacomp_rebuttal/")


for stuff in tqdm(all_stuff):

    num = len(s3.ls(stuff))

    input_shards = "pipe:aws s3 cp s3://" + str(stuff) + "/{0000000..00000" + str(num-1) + "}.tar -"

    pipeline = [
        wds.SimpleShardList(input_shards),
    ]

    pipeline.extend(
        [
            wds.split_by_worker,
            partial(tarfile_to_samples_nothrow, handler=wds.warn_and_stop),
            # wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.decode(
                # "pilrgb",
                partial(json_decoder, json_keys=["uid", "original_width", "original_height"]),
                detic_decoder,
                handler=wds.warn_and_stop,
            ),
            # wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.to_tuple("json", "classes", "boxes", "scores"),
            # wds.map_tuple(*unique_preprocessors),
            wds.batched(batch_size, partial=True),
        ]
    )

    loader = wds.WebLoader(
        wds.DataPipeline(*pipeline),
        batch_size=None,
        shuffle=False,
        num_workers=nworkers,
        persistent_workers=True,
    )


    parquet_dict = {
        "uid": [],
        "original_width": [],
        "original_height": [],
        "classes": [],
        "boxes": [],
        "scores": [],
    }
    for i, b in tqdm(enumerate(loader)):
        parquet_dict["uid"].extend([item[0] for item in b[0]])
        parquet_dict["original_width"].extend([item[1] for item in b[0]])
        parquet_dict["original_height"].extend([item[2] for item in b[0]])
        parquet_dict["classes"].extend(b[1])
        parquet_dict["boxes"].extend(b[2])
        parquet_dict["scores"].extend(b[3])
        # if i == 2:
        #     break
    name = stuff.split("/")[-1].split("_")[-1]
    pd.DataFrame.from_dict(parquet_dict).to_parquet(f's3://dcnlp-hub/datacomp_rebuttal_metadata2/{name}.parquet')
    # exit(0)