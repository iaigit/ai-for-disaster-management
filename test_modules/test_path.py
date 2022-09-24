def test_model_path(model_path: str) -> None:
    """Test if model_path is model path

    Parameters
    ----------
    model_path : str
        Model path in string
    """
    assert model_path.split(".")[-1] in ["pt", "pth", "ckpt"], ValueError(
        "Model path must be either in pt, pth, or ckpt format"
    )


def test_image_path(image_path: str) -> None:
    """Test if image_path is model path

    Parameters
    ----------
    image_path : str
        Model path in string
    """
    assert image_path.split(".")[-1].lower() in [
        "jpg",
        "jpeg",
        "png",
    ], ValueError("Model path must be either in jpg, jpeg, or png format")
