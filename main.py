import flash
from flash.image.detection import ObjectDetectionData, ObjectDetector


def build_od_model(
    head: str = "yolov5",
    backbone: str = "small",
    num_classes: int = 4,
    image_size: int = 640,
    dir_checkpoint: str = None,
) -> flash.Task:
    """Build object detection model using LightningFlash

    Parameters
    ----------
    head : str, optional
        head of the model, check ObjectDetector.available_heads() for more info, by default "yolov5"
    backbone : str, optional
        backbone of the model, check ObjectDetector.available_backbone() for more info, by default "small"
    num_classes : int, optional
        total class from annotations, by default 4
    image_size : int, optional
        image_size input of the model, by default 640
    dir_checkpoint : str, optional
        if model is already trained, pass this parameter to load the model, by default None

    Returns
    -------
    flash.Task
        Flash model
    """
    if dir_checkpoint is not None:
        assert dir_checkpoint.split(".")[-1] in [
            "pt",
            "pth",
            "ckpt",
        ], ValueError("Format file must be either pt, pth or ckpt")
    model = ObjectDetector(
        head=head,
        backbone=backbone,
        learning_rate=4e-3,
        num_classes=num_classes,
        image_size=640,
    )
    if dir_checkpoint:
        model = model.load_from_checkpoint(dir_checkpoint, precision=16)
    return model


def main() -> None:
    """Main function"""
    # transform = transforms.Compose(
    #     [transforms.RandomCrop(size=640), transforms.Normalize(0.5, 0.5)]
    # )
    # image = img_to_tensor("B01_0004.JPG", preprocessing=transform)
    model = build_od_model()
    trainer = flash.Trainer(resume_from_checkpoint="./model/last_ckpt.pt")
    datamodule = ObjectDetectionData.from_files(
        predict_files=[
            "B01_0004.JPG",
        ],
        transform_kwargs={"image_size": 640},
        batch_size=1,
    )
    predictions = trainer.predict(model, datamodule, output="preds")
    # predictions = model(image)
    print(predictions)


if __name__ == "__main__":
    main()
