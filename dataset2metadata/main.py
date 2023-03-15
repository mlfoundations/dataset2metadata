import os
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from pathlib import Path

import fire
import yaml
from dataloaders import create_loader
from registry import models_to_wrappers
from PIL import ImageFile
from utils import topsort

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.getLogger().setLevel(logging.INFO)

def check_yml(yml):

    # manditory fields in the yml
    yml_fileds = [
        'models',
        'postprocess_columns',
        'postprocess_features',
        'additional_fields',
        'nworkers',
        'batch_size',
        'device',
        'input_tars',
        'output_metadata_dir',
        'custom_models_pypath',
        'custom_postprocess_pypath',
    ]

    for f in yml_fileds:
        if f not in yml:
            raise ValueError(f'yml must contain field: {f}')

def process(
    yml_path,
):
    # parse yml and check resulting dict
    yml = yaml.safe_load(Path(yml_path).read_text())
    check_yml(yml)

    # create dataloader based on user input
    dataloader, input_map = create_loader(
        yml['input_tars'],
        yml['models'],
        yml['additional_fields'],
        yml['nworkers'],
        yml['batch_size'],
    )

    models = {m_str: models_to_wrappers[m_str](yml['device']) for m_str in yml['models']}
    topsort_order = topsort(
        {m_str: models_to_wrappers[m_str].dependencies for m_str in yml['models']}
    )

    logging.info(f'topsort order: {topsort_order}')

    for sample in dataloader:
        model_outputs = {}

        # eval all models sequentially in a top sort order
        for m_str in topsort_order:

            model_input = []
            cache = {}

            # fill the model input
            for i in input_map[m_str]:

                if isinstance(i, int):
                    if models[m_str].use_gpu and i not in cache:
                        cache[i] = sample[i].to(yml['device'])
                    else:
                        cache[i] = sample[i]

                    model_input.append(cache[i])
                else:
                    # use previously computed outputs ans new inputs
                    # NOTE: assume downstream model consumes on same device as upstream
                    assert i in model_outputs
                    model_input.append(model_outputs[i])

            with torch.no_grad():
                model_outputs[m_str] = models[m_str](*model_input)

def main():
    fire.Fire(process)

if __name__ == "__main__":
    main()