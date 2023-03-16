from functools import partial

import webdataset as wds
from preprocessors import json_decoder
from registry import model_lookup, preprocessor_lookup


def get_to_tuple_directives(models, additional_fields):

    wrapper_classes = [model_lookup[m] for m in models]

    input_map = {}

    # get unique preprocessor directive, which is a raw_input, preprocessor pair
    unique_derectives = []

    for i, wc in enumerate(wrapper_classes):
        assert len(wc.preprocessors) == len(wc.raw_inputs)

        preprocess_directives = [
            (wc.raw_inputs[i], wc.preprocessors[i]) for i in range(len(wc.preprocessors))
        ]

        input_map[models[i]] = []

        for j in range(len(preprocess_directives)):
            if preprocess_directives[j] not in unique_derectives:
                input_map[models[i]].append(len(unique_derectives))
                unique_derectives.append(preprocess_directives[j])
            else:
                input_map[models[i]].append(unique_derectives.index(preprocess_directives[j]))

        if len(wc.dependencies):
            # non-numeric, nameded dependencies, i.e., the outputs of other models
            input_map[models[i]].extend(wc.dependencies)

    # add directives to include data from the tars into the webdataset
    if additional_fields is not None and len(additional_fields):
        # NOTE: currently no support for these additional fields being taken as inputs to models
        input_map['json'] = [len(unique_derectives), ]
        unique_derectives.append(('json', 'identity'))

    return unique_derectives, input_map

def create_loader(input_shards, models, additional_fields, nworkers, batch_size):

    (
        unique_derectives,
        input_map,
    ) = get_to_tuple_directives(models, additional_fields)

    tuple_fields = [e[0] for e in unique_derectives]
    unique_preprocessors = [
        preprocessor_lookup[e[-1]] for e in unique_derectives
    ]

    pipeline = [wds.SimpleShardList(input_shards), ]

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