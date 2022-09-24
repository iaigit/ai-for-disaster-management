from collections import Counter, OrderedDict
from typing import Dict, List, Sequence, Tuple, Union

import flash
import numpy as np
import torch
import torchvision
from flash.image.detection import ObjectDetectionData, ObjectDetector
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from test_modules.test_path import test_image_path, test_model_path

SequenceOrTensor = Union[Sequence, torch.tensor]


def predict_from_trainer(
    trainer: flash.Task,
    model: torch.nn.Module,
    image_path: str,
    image_size: int = 640,
) -> SequenceOrTensor:
    """Predict using Trainer from flash

    Parameters
    ----------
    trainer : flash.Task
        trainer model from Lightning Flash
    model : torch.nn.Module
        model trained from Lightning Flash
    image_path : str
        Image path in jpg, jpeg, or png format
    image_size : int, optional
        Size of the image, by default 640
    Returns
    -------
    SequenceOrTensor
        Tensor contains of bounding boxes, class, and confidence
    """
    test_image_path(image_path)
    datamodule = ObjectDetectionData.from_files(
        predict_files=[
            image_path,
        ],
        transform_kwargs={"image_size": image_size},
        batch_size=1,
    )
    predictions = trainer.predict(model, datamodule, output="preds")

    return predictions


def img_to_tensor(
    img_path: str, preprocessing: torchvision.transforms = None
) -> SequenceOrTensor:
    """Read image to turn intu Tensor, add preprocessing if neccesary

    Parameters
    ----------
    img_path : str
        Image path of the data
    preprocessing : torchvision.transforms, optional
        Add transformation, by default None

    Returns
    -------
    SequenceOrTensor
        Tensor with RGB dimension
    """
    assert img_path.split(".")[-1].lower() in [
        "jpg",
        "png",
        "jpeg",
    ], ValueError("Format file must be either jpg, png, or jpeg")
    image = read_image(img_path)
    image = image.type(torch.float16)
    image = image / 255.0
    if preprocessing:
        image = preprocessing(image)
    return image[None]


def load_flash_model(
    model_path: str, **trainer_kwargs
) -> Tuple[flash.Task, torch.nn.Module]:
    """Load flash model by passing model_path

    Parameters
    ----------
    model_path : str
        path model with the pt, pth or ckpt format
    **trainer_kwargs : dict, optional
        Keyword arguments for Trainer LightningFlash

    Returns
    -------
    Tuple[flash.Task, torch.nn.Module]
        Return trainer and model
    """
    test_model_path(model_path)
    model = ObjectDetector(
        head="yolov5",
        backbone="small",
        learning_rate=4e-3,
        num_classes=4,
        image_size=640,
    )
    model = model.load_from_checkpoint(model_path)
    trainer = flash.Trainer(
        resume_from_checkpoint=model_path,
        enable_checkpointing=False,
        precision=32,
        **trainer_kwargs,
    )
    return trainer, model


def plot_img_with_bbox(image_path: str, result_annotations: List) -> Image:
    """Generate image with bounding box for predictions

    Parameters
    ----------
    image_path : str
        Image path of the prediction with the format jpg, png, or jpeg
    result_annotations : List
        Result annotations from the LightningFlash model

    Returns
    -------
    Image
        Image that has the bounding box in it
    """
    test_image_path(image_path)
    result_annotations = result_annotations[0][0]
    list_bbox = []
    for bbox in result_annotations["bboxes"]:
        temp_bbox = []
        temp_bbox.append(int(bbox["xmin"]))
        temp_bbox.append(int(bbox["ymin"]))
        temp_bbox.append(int(bbox["xmin"]) + int(bbox["width"]))
        temp_bbox.append(int(bbox["ymin"]) + int(bbox["height"]))
        list_bbox.append(temp_bbox)

    labels = ["H", "LD", "HD", "other"]
    list_class = result_annotations["labels"]
    list_class = [labels[x - 1] for x in list_class]
    # list_scores = result_annotations["scores"]
    sample_image = Image.open(image_path)
    tensor_image = (transforms.ToTensor()(sample_image) * 255).to(torch.uint8)
    tensor_annotations = torch.tensor(list_bbox)
    dict_color = {"other": "black", "LD": "yellow", "H": "green", "HD": "red"}
    colors = [dict_color[x] for x in list_class]
    img = draw_bounding_boxes(
        tensor_image,
        boxes=tensor_annotations,
        labels=list_class,
        width=3,
        colors=colors,
    )
    view_image = transforms.ToPILImage()(img)
    return view_image


def get_info_predictions(predictions: List) -> Tuple[int, Dict, float]:
    """Get summary information from predictions
    The information is Total Bounding boxes, Each class predictions, and average scores

    Parameters
    ----------
    predictions : List
        Result predictions from LightningFlash

    Returns
    -------
    Tuple[int, Dict, float]
        Tuple of Total Bounding boxes, Each class predictions, and average scores
    """
    predictions = predictions[0][0]
    total_bboxes = len(predictions["bboxes"])
    labels = ["H", "LD", "HD", "other"]
    dict_labels = OrderedDict({"H": 0, "LD": 0, "HD": 0, "other": 0})
    label_list = predictions["labels"]
    final_list = [labels[x - 1] for x in label_list]
    dict_labels.update(Counter(final_list))
    avg_scores = sum(predictions["scores"]) / len(predictions["scores"])
    return total_bboxes, dict(dict_labels), avg_scores


def plot_img_with_bbox_v2(
    image_array: np.ndarray, result_annotations: List
) -> Image:
    """Generate image with bounding box for predictions but the input is numpy array

    Parameters
    ----------
    image_array : np.ndarray
        Image array of the prediction with the type of np.ndarray
    result_annotations : List
        Result annotations from the LightningFlash model

    Returns
    -------
    Image
        Image that has the bounding box in it
    """
    assert type(image_array) == np.ndarray, ValueError(
        "Type of array is not Numpy"
    )
    result_annotations = result_annotations[0][0]
    list_bbox = []
    for bbox in result_annotations["bboxes"]:
        temp_bbox = []
        temp_bbox.append(int(bbox["xmin"]))
        temp_bbox.append(int(bbox["ymin"]))
        temp_bbox.append(int(bbox["xmin"]) + int(bbox["width"]))
        temp_bbox.append(int(bbox["ymin"]) + int(bbox["height"]))
        list_bbox.append(temp_bbox)

    labels = ["H", "LD", "HD", "other"]
    list_class = result_annotations["labels"]
    list_class = [labels[x - 1] for x in list_class]
    # list_scores = result_annotations["scores"]
    tensor_image = torch.tensor(image_array)
    tensor_image = tensor_image.to(torch.uint8)
    tensor_annotations = torch.tensor(list_bbox)
    dict_color = {"other": "black", "LD": "yellow", "H": "green", "HD": "red"}
    colors = [dict_color[x] for x in list_class]
    img = draw_bounding_boxes(
        tensor_image,
        boxes=tensor_annotations,
        labels=list_class,
        width=3,
        colors=colors,
    )
    view_image = transforms.ToPILImage()(img)
    return view_image
