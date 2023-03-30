from functools import partial

import torch
import torch.nn as nn

from dataset2metadata.postprocessors import identity
from open_clip import create_model_and_transforms


# define model here
class OpenClipWrapper(nn.Module):

    name = 'open-clip-vit-l14'
    raw_inputs = ['image', 'text']
    preprocessors = ['clip-aug', 'clip-tokens']
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()

        checkpoint_path = None
        self.model, _, _ = create_model_and_transforms('ViT-L-14', pretrained=checkpoint_path, device=device)
        self.model.eval()

        print(f'instantiated {self.name} on {device}')

    def forward(self, x):
        generated_ids = self.model.generate(pixel_values=x)
        generated_text = [
            t.strip() for t in bp.batch_decode(generated_ids, skip_special_tokens=True)
        ]

        return generated_text

# define model loopup
model_lookup = {
    'open-clip-vit-l14': OpenClipWrapper,
}

