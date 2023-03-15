import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from pathlib import Path

import fire
import yaml
from dataloaders import create_wd_loader
from registry import models_to_wrappers
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.getLogger().setLevel(logging.INFO)

def check_yml_dict(yml_dict):

    # manditory fields in the yml
    yml_fileds = [
        'models',
        'postprocess',
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
        if f not in yml_dict:
            raise ValueError(f'yml must contain field: {f}')

def process(
    yml_path,
):
    # parse yml and check resulting dict
    yml_dict = yaml.safe_load(Path(yml_path).read_text())
    check_yml_dict(yml_dict)

    # create dataloader based on user input
    dataloader, input_map = create_wd_loader(
        yml_dict['input_tars'],
        yml_dict['models'],
        yml_dict['additional_fields'],
        yml_dict['nworkers'],
        yml_dict['batch_size'],
    )

    models = [models_to_wrappers[m](yml_dict['device']) for m in yml_dict['models']]

    print(input_map)

    for sample in dataloader:
        # print(len(sample))
        exit(0)
        pass

def main():
    fire.Fire(process)

if __name__ == "__main__":
    main()