from functools import partial
import logging
import hashlib
import json

import webdataset as wds
from dataset2metadata.preprocessors import json_decoder
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)

logging.getLogger().setLevel(logging.INFO)


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    # taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)

    return samples


def add_json(sample):
    if "json" not in sample:
        # sample.pop("npy")
        sample["json"] = json.dumps(
            {
                "url": sample["__url__"],
                "key": sample["__key__"],
                "uid": hashlib.md5(str(sample["__key__"]).encode()).hexdigest(),
            }
        ).encode("utf-8")
    return sample


def filter_no_caption_or_no_image(sample):
    # taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def get_to_tuple_directives(models, additional_fields):
    # import here as registry may have updated
    from dataset2metadata.registry import model_lookup

    wrapper_classes = [model_lookup[m] for m in models]

    input_map = {}

    # get unique preprocessor directive, which is a raw_input, preprocessor pair
    unique_derectives = []

    for i, model_class in enumerate(wrapper_classes):
        assert len(model_class.preprocessors) == len(model_class.raw_inputs)

        preprocess_directives = [
            (model_class.raw_inputs[k], model_class.preprocessors[k])
            for k in range(len(model_class.preprocessors))
        ]

        input_map[models[i]] = []

        for j in range(len(preprocess_directives)):
            if preprocess_directives[j] not in unique_derectives:
                input_map[models[i]].append(len(unique_derectives))
                unique_derectives.append(preprocess_directives[j])
            else:
                input_map[models[i]].append(
                    unique_derectives.index(preprocess_directives[j])
                )

        if len(model_class.dependencies):
            # non-numeric, nameded dependencies, i.e., the outputs of other models
            input_map[models[i]].extend(model_class.dependencies)

    # add directives to include data from the tars into the webdataset
    if additional_fields is not None and len(additional_fields):
        # NOTE: currently no support for these additional fields being taken as inputs to models
        input_map["json"] = [
            len(unique_derectives),
        ]
        unique_derectives.append(("json", "identity"))

    return unique_derectives, input_map


def create_loader(input_shards, models, additional_fields, nworkers, batch_size):
    # import here as registry may have updated
    from dataset2metadata.registry import preprocessor_lookup

    (
        unique_derectives,
        input_map,
    ) = get_to_tuple_directives(models, additional_fields)

    tuple_fields = [e[0] for e in unique_derectives]
    unique_preprocessors = [preprocessor_lookup[e[-1]] for e in unique_derectives]

    logging.info(input_shards)
    pipeline = [
        wds.SimpleShardList(input_shards),
    ]

    pipeline.extend(
        [
            wds.split_by_worker,
            tarfile_to_samples_nothrow,
            wds.map(add_json),
            # wds.select(filter_no_caption_or_no_image),
            wds.decode(
                "pilrgb",
                partial(json_decoder, json_keys=additional_fields),
                # handler=wds.warn_and_continue,
            ),
            wds.rename(image="jpg;png;jpeg;webp"),  # , text="txt"),
            wds.to_tuple(*tuple_fields),
            wds.map_tuple(*unique_preprocessors),
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

    return loader, input_map
