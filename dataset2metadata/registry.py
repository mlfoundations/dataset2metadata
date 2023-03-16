
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
    'oai-clip-vit-b32-score': partial(post.batched_dot_product, a='oai-clip-vit-b32'),
    'oai-clip-vit-l14-score': partial(post.batched_dot_product, a='oai-clip-vit-l14'),
    'nsfw-detoxify-score': partial(post.identity, a='nsfw-detoxify'),
    'nsfw-image-score': partial(post.identity, a='nsfw-image'),
    'dedup-isc-ft-v107-score': partial(post.select, a='dedup-isc-ft-v107', index=1),
    'json-identity': partial(post.identity, a='json'), # TODO
}

postprocess_feature_lookup = {
    'oai-clip-vit-b32-img': partial(post.select, a='oai-clip-vit-b32', index=0),
    'oai-clip-vit-b32-txt': partial(post.select, a='oai-clip-vit-b32', index=1),
    'oai-clip-vit-l14-img': partial(post.select, a='oai-clip-vit-l14', index=0),
    'oai-clip-vit-l14-txt': partial(post.select, a='oai-clip-vit-l14', index=1),
    'dedup-isc-ft-v107-img': partial(post.select, a='dedup-isc-ft-v107', index=0),
}
