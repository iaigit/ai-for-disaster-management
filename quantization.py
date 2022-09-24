import torch
from flash.image.detection import ObjectDetector


def main() -> None:
    """Main model"""
    model = ObjectDetector.load_from_checkpoint("./model/last_ckpt.pt")
    model_int8 = torch.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8,
    )
    torch.save(model_int8.state_dict(), "model_quantizied.pth")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
