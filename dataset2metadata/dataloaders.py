from functools import partial
import webdataset as wds
from registry import models_to_wrappers
from preprocessors import json_decoder


def get_to_tuple_directives(models, additional_fields):

    wrapper_classes = [models_to_wrappers[m] for m in models]

    input_map = {}

    # get unique preprocessors
    unique_preprocessors = []
    tuple_fields = []
    for i, wc in enumerate(wrapper_classes):
        if wc.preprocessors is not None:
            input_map[models[i]] = []
            for j in range(len(wc.preprocessors)):
                if wc.preprocessors[j] not in unique_preprocessors:
                    input_map[models[i]].append(len(tuple_fields))
                    unique_preprocessors.append(wc.preprocessors[j])
                    tuple_fields.append(wc.raw_inputs[j])
                else:
                    input_map[models[i]].append(unique_preprocessors.index(wc.preprocessors[j]))

    # add directives to include data from the tars into the webdataset
    if additional_fields is not None and len(additional_fields):
        input_map['json'] = [len(tuple_fields), ]
        unique_preprocessors.append(lambda x: x)
        tuple_fields.append('json')

    return tuple_fields, unique_preprocessors, input_map

def create_wd_loader(input_shards, models, additional_fields, nworkers, batch_size):

    (
        tuple_fields,
        unique_preprocessors,
        input_map,
    ) = get_to_tuple_directives(models, additional_fields)

    pipeline = [wds.SimpleShardList(input_shards)]

    pipeline.extend([
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode(
            'pilrgb',
            partial(json_decoder, json_keys=additional_fields),
            handler=wds.warn_and_continue),
        wds.rename(image='jpg;png;jpeg;webp', text='txt'),
        wds.to_tuple(*tuple_fields),
        wds.map_tuple(*unique_preprocessors),
        wds.batched(batch_size, partial=True),
    ])

    loader = wds.WebLoader(
        wds.DataPipeline(*pipeline),
        batch_size=None,
        shuffle=False,
        num_workers=nworkers,
        persistent_workers=True,
    )

    return loader, input_map