import logging
from typing import List

import pandas as pd
import numpy as np

logging.getLogger().setLevel(logging.INFO)

class Writer(object):

    def __init__(
            self,
            name: str,
            postprocess_feature_fields: List[str],
            postprocess_parquet_fields: List[str],
            additional_columns: List[str]
        ) -> None:
        self.name = name
        self.feature_store = {e : [] for e in postprocess_feature_fields}
        self.parquet_store = {e: [] for e in postprocess_parquet_fields + additional_columns}

    def update_feature_store(self, k, v):
        self.feature_store[k].append(v)

    def update_parquet_store(self, k, v):
        self.parquet_store[k].append(v)

    def write(out_dir_path, self):
        # TODO
        try:
            df = pd.DataFrame.from_dict(self.get_parquet_dict())
            df.to_parquet(f'{out_dir_path}/{self.name}.parquet')
            np.savez_compressed(f'{out_dir_path}/{self.name}.npz')

            return True

        except Exception as e:
            logging.exception(e)
            logging.error(f'failed to write metadata for shard: {self.shard_name}')
            return False