# custom.py implementation

from functools import partial

import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from dataset2metadata.postprocessors import identity
from clip import clip

hf_preprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# define model here
class Blip2Wrapper(nn.Module):

    name = 'blip2' # static field: provide name of model
    raw_inputs = ['image', ] # static field: provide name of input to process
    preprocessors = ['blip2-aug', ] # static field: name of preprocessor
    dependencies = [] # static field: other models that should be evaluated before this model
    to_device = True # static field: if True, move input to device

    def __init__(self, device) -> None:
        super().__init__()
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16).to(device)
        self.model.config.text_config.min_length = 5
        self.model.config.text_config.max_length = 40

        self.model.config.text_config.do_sample = True
        self.model.config.text_config.top_k = 50
        self.model.config.text_config.temperature = 0.75

        self.model.eval()
        print(f'instantiated {self.name} on {device}')

    def forward(self, x):
        generated_ids = self.model.generate(pixel_values=x)
        generated_text = [
            t.strip() for t in hf_preprocessor.batch_decode(generated_ids, skip_special_tokens=True)
        ]

        return generated_text

def blip_preprocess(x):
    # custom preprocessor function

    if x.height < 5:
        new_width = int(x.width * 5 / x.height)
        x = x.resize((new_width, 5))

    if x.width < 5:
        new_height = int(x.height * 5 / x.width)
        x = x.resize((5, new_height))

    a = hf_preprocessor(images=x, return_tensors="pt").to(torch.float16)

    return a['pixel_values'].squeeze()

# map preprocessor strings to preprocessor functions
preprocessor_lookup = {
    'blip2-aug': blip_preprocess,
}

# map model strings to model classes
model_lookup = {
    'blip2': Blip2Wrapper,
}

# map postprocessor strings to functions, outputs saved to column-store parquet
postprocess_parquet_lookup = {
    'blip2-cap': partial(identity, model='blip2', to_cpu=False),
}
# map postprocessor strings to functions, outputs saved to feature store npz file
postprocess_feature_lookup = {}