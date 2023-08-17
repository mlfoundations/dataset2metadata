# custom.py implementation

from functools import partial

import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from dataset2metadata.postprocessors import select, batched_dot_product_index
from clip import clip

hf_preprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")


def blip_preprocess(x):
    # custom preprocessor function

    if x.height < 5:
        new_width = int(x.width * 5 / x.height)
        x = x.resize((new_width, 5))

    if x.width < 5:
        new_height = int(x.height * 5 / x.width)
        x = x.resize((5, new_height))

    a = hf_preprocessor(images=x, return_tensors="pt").to(torch.float16)

    return a["pixel_values"].squeeze()


class Blip2ClipB32L14Wrapper(nn.Module):
    name = "blip2clipb32l14"  # static field: provide name of model
    raw_inputs = [
        "image",
        "image",
        "text",
    ]  # static field: provide name of input to process
    preprocessors = [
        "blip2-aug",
        "clip-aug",
        "clip-tokens",
    ]  # static field: name of preprocessor
    dependencies = (
        []
    )  # static field: other models that should be evaluated before this model
    to_device = True  # static field: if True, move input to device

    def __init__(self, device) -> None:
        super().__init__()
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(device)
        self.blip2.config.text_config.min_length = 5
        self.blip2.config.text_config.max_length = 40

        self.blip2.config.text_config.do_sample = True
        self.blip2.config.text_config.top_k = 50
        self.blip2.config.text_config.temperature = 0.75

        self.blip2.eval()

        self.b32, _ = clip.load("ViT-B/32", device=device)
        self.b32.eval()

        self.l14, _ = clip.load("ViT-L/14", device=device)
        self.l14.eval()
        self.device = device

        print(f"instantiated {self.name} on {device}")

    def forward(self, x_blip, x_clip, t_clip):
        generated_ids = self.blip2.generate(pixel_values=x_blip)
        generated_text = [
            t.strip()
            for t in hf_preprocessor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        ]

        # tokenize generated text
        t_blip2 = clip.tokenize(generated_text).to(self.device)

        b32_image_feature = self.b32.encode_image(x_clip)
        b32_text_feature = self.b32.encode_text(t_clip)
        b32_generated_text_feature = self.b32.encode_text(t_blip2)

        b32_image_feature = b32_image_feature / b32_image_feature.norm(
            dim=1, keepdim=True
        )
        b32_text_feature = b32_text_feature / b32_text_feature.norm(dim=1, keepdim=True)
        b32_generated_text_feature = (
            b32_generated_text_feature
            / b32_generated_text_feature.norm(dim=1, keepdim=True)
        )

        l14_image_feature = self.l14.encode_image(x_clip)
        l14_text_feature = self.l14.encode_text(t_clip)
        l14_generated_text_feature = self.l14.encode_text(t_blip2)

        l14_image_feature = l14_image_feature / l14_image_feature.norm(
            dim=1, keepdim=True
        )
        l14_text_feature = l14_text_feature / l14_text_feature.norm(dim=1, keepdim=True)
        l14_generated_text_feature = (
            l14_generated_text_feature
            / l14_generated_text_feature.norm(dim=1, keepdim=True)
        )

        return (
            b32_image_feature,
            b32_text_feature,
            b32_generated_text_feature,
            l14_image_feature,
            l14_text_feature,
            l14_generated_text_feature,
            generated_text,
        )


# map preprocessor strings to preprocessor functions
preprocessor_lookup = {
    "blip2-aug": blip_preprocess,
}

# map model strings to model classes
model_lookup = {
    "blip2clipb32l14": Blip2ClipB32L14Wrapper,
}

# map postprocessor strings to functions, outputs saved to column-store parquet
postprocess_parquet_lookup = {
    "blip2-cap": partial(select, model="blip2clipb32l14", index=6, to_cpu=False),
    "oai-clip-b32-score": partial(
        batched_dot_product_index, i=0, j=1, model="blip2clipb32l14"
    ),
    "oai-clip-b32-blip2-score": partial(
        batched_dot_product_index, i=0, j=2, model="blip2clipb32l14"
    ),
    "oai-clip-l14-score": partial(
        batched_dot_product_index, i=3, j=4, model="blip2clipb32l14"
    ),
    "oai-clip-l14-blip2-score": partial(
        batched_dot_product_index, i=3, j=5, model="blip2clipb32l14"
    ),
}

postprocess_feature_lookup = {
    "oai-clip-b32-image": partial(select, model="blip2clipb32l14", index=0),
    "oai-clip-b32-text": partial(select, model="blip2clipb32l14", index=1),
    "oai-clip-b32-blip2-text": partial(select, model="blip2clipb32l14", index=2),
    "oai-clip-l14-image": partial(select, model="blip2clipb32l14", index=3),
    "oai-clip-l14-text": partial(select, model="blip2clipb32l14", index=4),
    "oai-clip-l14-blip2-text": partial(select, model="blip2clipb32l14", index=5),
}
