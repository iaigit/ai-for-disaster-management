from utils import load_flash_model


def main() -> None:
    """Main model"""
    trainer, model = load_flash_model("./model/last_ckpt.pt")
    model.predict_kwargs = {}
    model.serve()


if __name__ == "__main__":
    main()
