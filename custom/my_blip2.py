from functools import partial

import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2Processor, Blip2Config

from dataset2metadata.postprocessors import identity

bp = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#config: https://huggingface.co/Salesforce/blip2-opt-2.7b/resolve/main/config.json

def blip_pre(x):
    '''
    if x.height == 1 and x.width == 1:
        # edge case as huggingface tries to guess the channel dim
        x = x.resize((2, 2))

    if x.height == 3 and x.width == 3:
        # edge case as huggingface tries to guess the channel dim
        x = x.resize((4, 4))
    '''
    if x.height < 5:
        new_width = int(x.width * 5 / x.height)
        x = x.resize((new_width, 5))

    if x.width < 5:
        new_height = int(x.height * 5 / x.width)
        x = x.resize((5, new_height))

    a = bp(images=x, return_tensors="pt").to(torch.float16)

    return a['pixel_values'].squeeze()

# define model here
class Blip2Wrapper(nn.Module):

    name = 'blip2'
    raw_inputs = ['image', ]
    preprocessors = ['blip2-aug', ]
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16).to(device)

        self.model.eval()
        self.model.config.text_config.min_length = 5
        self.model.config.text_config.max_length = 40
        #self.model.config.text_config.do_sample = True
        #self.model.config.text_config.top_p = 0.9
        #self.model.config.text_config.repetition_penality = 1.1
        print(f'instantiated {self.name} on {device}')

    def forward(self, x):
        generated_ids = self.model.generate(pixel_values=x)
        generated_text = [
            t.strip() for t in bp.batch_decode(generated_ids, skip_special_tokens=True)
        ]
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
    'blip2-cap': partial(identity, model='blip2', to_cpu=False),
}
postprocess_feature_lookup = {}
