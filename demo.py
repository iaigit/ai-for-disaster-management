import gradio as gr
import numpy as np
from flash.image.detection import ObjectDetectionData

from utils import get_info_predictions, load_flash_model, plot_img_with_bbox_v2

trainer, model = load_flash_model("./model/last_ckpt.pt")


def upload_image(input_image: np.ndarray) -> np.ndarray:
    """Generate function for Gradio

    Parameters
    ----------
    input_image : np.ndarray
        input that is given from Gradio interface

    Returns
    -------
    np.ndarray
        numpy array that will be given back to Gradio interface
    """
    input_image = input_image.transpose(2, 0, 1)
    datamodule = ObjectDetectionData.from_numpy(
        predict_data=[input_image],
        transform_kwargs={"image_size": 640},
        batch_size=1,
    )
    model.predict_kwargs = {}
    trainer.enable_checkpointing = False
    predictions = trainer.predict(model, datamodule, output="preds")
    result_image = plot_img_with_bbox_v2(input_image, predictions)
    total_bboxes, dict_labels, avg_scores = get_info_predictions(predictions)
    return np.array(result_image), total_bboxes, avg_scores, dict_labels


def main() -> None:
    """Main function"""
    title = "Forest Damage Detection"
    description = """
<center>
Forest Damage Detection with 4 labels: H (Healthy), LD (Low Damage), HD (High Damage), Other
</center>
    """
    demo = gr.Interface(
        fn=upload_image,
        inputs=gr.Image(shape=(640, 640)),
        outputs=[
            "image",
            gr.Textbox(lines=1, label="Total Trees"),
            gr.Textbox(
                lines=1, label="Average Confidence of each Bounding Box"
            ),
            gr.Textbox(lines=1, label="Total Trees for each label"),
        ],
        title=title,
        description=description,
    )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
