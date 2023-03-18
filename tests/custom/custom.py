from functools import partial

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch.nn as nn
import torch

bp = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

def identity(cache, k):
    return cache[k]

def blip_pre(x):

    if x.height == 1 and x.width == 1:
        # edge case as huggingface tries to guess the channel dim
        x = x.resize((2, 2))

    if x.height == 3 and x.width == 3:
        # edge case as huggingface tries to guess the channel dim
        x = x.resize((6, 6))

    a = bp(images=x, return_tensors="pt").to(torch.float16)

    return a['pixel_values'].squeeze()

# define model here
class Blip2Wrapper(nn.Module):

    name = 'blip2'
    raw_inputs = ['image', ]
    preprocessors = ['blip2-aug', ]
    dependencies = []
    use_gpu = True

    def __init__(self, device) -> None:
        super().__init__()
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16).to(device)

        self.model.eval()
        print(f'instantiated {self.name} on {device}')

    def forward(self, x):
        generated_ids = self.model.generate(pixel_values=x)
        generated_text = [t.strip() for t in bp.batch_decode(generated_ids, skip_special_tokens=True)]

        return generated_text


# define preprocessor map
preprocessor_lookup = {
    'blip2-aug': blip_pre,
}

# define model loopup
model_lookup = {
    'blip2': Blip2Wrapper,
}

# postprocessors
postprocess_parquet_lookup = {
    'blip2-cap': partial(identity, k='blip2'),
}
postprocess_feature_lookup = {}