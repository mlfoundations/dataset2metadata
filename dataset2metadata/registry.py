
import inspect
from functools import partial

import models
import postprocessors as post
import preprocessors as pre

model_lookup = {
    cls.name: cls for _, cls in models.__dict__.items() if hasattr(cls, 'name') and inspect.isclass(cls)
}

preprocessor_lookup = {
    'clip-aug': pre.oai_clip_image,
    'clip-tokens': pre.oai_clip_text,
    'identity': pre.identity,
    'dedup-aug': pre.dedup,
    'faces-aug': pre.faces_scrfd,
}

# Postprocessors
postprocess_parquet_lookup = {
    'oai-clip-vit-b32-score': partial(post.batched_dot_product, k='oai-clip-vit-b32'),
    'oai-clip-vit-l14-score': partial(post.batched_dot_product, k='oai-clip-vit-l14'),
    'nsfw-detoxify-score': partial(post.identity, k='nsfw-detoxify'),
    'nsfw-image-score': partial(post.identity, k='nsfw-image-oai-clip-vit-l-14'),
    'dedup-isc-ft-v107-score': partial(post.select, k='dedup-isc-ft-v107', index=1),
    'json-transpose': partial(post.transpose_list, k='json'),
}

postprocess_feature_lookup = {
    'oai-clip-vit-b32-image': partial(post.select, k='oai-clip-vit-b32', index=0),
    'oai-clip-vit-b32-text': partial(post.select, k='oai-clip-vit-b32', index=1),
    'oai-clip-vit-l14-image': partial(post.select, k='oai-clip-vit-l14', index=0),
    'oai-clip-vit-l14-text': partial(post.select, k='oai-clip-vit-l14', index=1),
    'dedup-isc-ft-v107-image': partial(post.select, k='dedup-isc-ft-v107', index=0),
}
