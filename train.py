from flash.image.detection import ObjectDetectionData

from test_modules.test_path import test_image_path, test_model_path
from utils import (
    SequenceOrTensor,
    img_to_tensor,
    load_flash_model,
    plot_img_with_bbox,
)


def predict_from_trainer(
    model_path: str, image_path: str, image_size: int = 640, **trainer_kwargs
) -> SequenceOrTensor:
    """Predict using Trainer from flash

    Parameters
    ----------
    model_path : str
        Model path in ckpt, pt, or pth format
    image_path : str
        Image path in jpg, jpeg, or png format
    image_size : int, optional
        Size of the image, by default 640
    **trainer_kwargs : dict, optional
        Keyword arguments for Trainer

    Returns
    -------
    SequenceOrTensor
        Tensor contains of bounding boxes, class, and confidence
    """
    test_model_path(model_path)
    test_image_path(image_path)

    trainer, model = load_flash_model(model_path, **trainer_kwargs)
    datamodule = ObjectDetectionData.from_files(
        predict_files=[
            image_path,
        ],
        transform_kwargs={"image_size": image_size},
        batch_size=1,
    )
    predictions = trainer.predict(model, datamodule, output="preds")

    return predictions


def predict_from_model(
    model_path: str, image_path: str, image_size: int = 640
) -> SequenceOrTensor:
    """Predict using Trainer from flash

    Parameters
    ----------
    model_path : str
        Model path in ckpt, pt, or pth format
    image_path : str
        Image path in jpg, jpeg, or png format
    image_size : int, optional
        Size of the image, by default 640

    Returns
    -------
    SequenceOrTensor
        Tensor contains of bounding boxes, class, and confidence
    """
    test_model_path(model_path)
    test_image_path(image_path)

    _, model = load_flash_model(
        model_path, **{"precision": 32, "enable_checkpointing": False}
    )
    image = img_to_tensor(image_path)

    predictions = model(image)
    return predictions


def main() -> None:
    predictions = predict_from_trainer("./last_ckpt.pt", "sample_image.jpg")
    image = plot_img_with_bbox("./sample_image.jpg", predictions)
    image.save("example_result.jpg")


if __name__ == "__main__":
    main()
