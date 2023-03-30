from importlib.machinery import SourceFileLoader
import os

import torch
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import logging
from pathlib import Path

import yaml
import fsspec
from dataset2metadata.dataloaders import create_loader
from PIL import ImageFile
from dataset2metadata.registry import update_registry
from dataset2metadata.utils import topsort
from dataset2metadata.writer import Writer

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
        'custom_pypath',
    ]

    for f in yml_fileds:
        if f not in yml:
            raise ValueError(f'yml must contain field: {f}')

def process(
    yml,
):

    if type(yml) is str:
        # parse yml and check resulting dict
        yml = yaml.safe_load(Path(yml).read_text())

    check_yml(yml)

    # if local out dir does not exist make it
    fs, output_path = fsspec.core.url_to_fs(yml['output_metadata_dir'])
    fs.makedirs(output_path, exist_ok=True)

    # if the user specifies specific custom implementaion of their own update the registry
    if yml['custom_pypath'] is not None:
        custom = SourceFileLoader(
            pathlib.Path(yml['custom_pypath']).stem,
            yml['custom_pypath']
        ).load_module()

        update_registry(custom)

    # import from registry here after we have updated
    from dataset2metadata.registry import (
        model_lookup, postprocess_feature_lookup,
        postprocess_parquet_lookup
    )

    # create dataloader based on user input
    dataloader, input_map = create_loader(
        yml['input_tars'],
        yml['models'],
        yml['additional_fields'],
        yml['nworkers'],
        yml['batch_size'],
    )

    # initializing models
    models = {m_str: model_lookup[m_str](yml['device']) for m_str in yml['models']}

    # deciding order to run them in based on dependencies
    topsort_order = topsort(
        {m_str: model_lookup[m_str].dependencies for m_str in yml['models']}
    )

    logging.info(f'topsort model evaluation order: {topsort_order}')

    # initialize the writer that stores results and dumps them to store
    # TODO: fix the name here
    writer = Writer(
        hashlib.md5(str(yml['input_tars']).encode()).hexdigest(),
        yml['postprocess_features'],
        yml['postprocess_columns'] + yml['additional_fields'],
    )

    for sample in dataloader:
        model_outputs = {}

        # eval all models sequentially in a top sort order
        for m_str in topsort_order:

            model_input = []
            cache = {}

            # fill the model input
            for i in input_map[m_str]:

                if isinstance(i, int):
                    if models[m_str].to_device and i not in cache:
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

            # TODO: make this more general, right now assumes last entry is json fields
            if len(yml['additional_fields']):
                model_outputs['json'] = sample[-1]

        for k in yml['postprocess_features']:
            writer.update_feature_store(k, postprocess_feature_lookup[k](model_outputs))

        for k in yml['postprocess_columns']:
            writer.update_parquet_store(k, postprocess_parquet_lookup[k](model_outputs))

        # if additional fields from json need to be saved, add those to the store
        if len(yml['additional_fields']):
            transposed_additional_fields = postprocess_parquet_lookup['json-transpose'](model_outputs)
            assert len(transposed_additional_fields) == len(yml['additional_fields'])
            for i, v in enumerate(transposed_additional_fields):
                writer.update_parquet_store(yml['additional_fields'][i], v)

    writer.write(yml['output_metadata_dir'])
