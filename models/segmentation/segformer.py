import torch
from typing import Optional, Union, Tuple, Any
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from functools import partial
import torchvision.transforms.functional as F
from PIL import Image


class FireSegmenter():
    def __init__(self,
                 size: Union[int, Tuple[int, int]] = (1280, 720),
                 pretrained_path: Any = './weights/segmentation/segformer-b0-segments-flame',
                 ) -> None:
        super().__init__()
        self.size = size
        self.input_fn = partial(SegformerFeatureExtractor(size=size),
                                return_tensors="pt")
        self.predict_fn = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_path)

    def output_fn(self,
                  logits: torch.FloatTensor,
                  image_size: Tuple[int, int]
                  ) -> torch.IntTensor:
        # First, rescale logits to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_size[-2:],
            mode='bilinear',
            align_corners=False
        )

        # Second, apply argmax on the class dimension
        return upsampled_logits.argmax(dim=1)[0]

    def __call__(self,
                 image: Union[Image.Image, torch.FloatTensor]
                 ) -> torch.IntTensor:
        _image = image
        if isinstance(image, Image.Image):
            _image = F.to_tensor(image)
        inputs = self.input_fn(image)
        predictions = self.predict_fn(inputs['pixel_values'])
        outputs = self.output_fn(predictions.logits, _image.size())

        return outputs
