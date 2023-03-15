
import inspect

import models
import postprocessors as post


models_to_wrappers = {
    cls.name: cls for _, cls in models.__dict__.items() if hasattr(cls, 'name') and inspect.isclass(cls)
}

# Postprocessors
postprocess_columnn_store_to_postprocessors = {
    'oai-clip-vit-b32-score': post.batched_dot_product,
    'oai-clip-vit-l14-score': post.batched_dot_product,
    'nsfw-detoxify-score': None,
    'nsfw-image-score': None,
    'dedup-isc-ft-v107-score': None,
    'json-identity': None,
}

# postprocess_features_store_to_postprocessors = {
#     'oai-clip-vit-b32-img': partial(post.select, index=0),
#     'oai-clip-vit-b32-txt': partial(post.select, index=1),
#     'oai-clip-vit-l14-img': partial(post.select, index=0),
#     'oai-clip-vit-l14-txt': partial(post.select, index=1),
#     'dedup-isc-ft-v107-img': None,
# }
