import numpy as np
import albumentations as A

THRESHOLD = 0.5
IMAGE_INPUT_SIZE = 512
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float16)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float16)


def get_output_model(image, ort_session):
    predict = ort_session.get_inputs()[0].name
    predict = ort_session.run(
        None,
        {predict: image},
    )
    return predict


def preprocessing_image(image):
    image_augmentation = A.Compose(
        [
            A.Normalize(MEAN, STD),
        ]
    )
    image = image_augmentation(image=image)["image"]
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return image


def postprocessing_image(predict):
    predict = predict[0][0][1]
    predict = np.where(predict > THRESHOLD, predict, 0)
    predict = (predict * 255).astype("uint8")
    return predict


def predict_image(image, ort_session):
    image = np.array(image.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)).convert("RGB"))
    input_image = preprocessing_image(image)
    predict = get_output_model(input_image, ort_session)
    predict = postprocessing_image(predict)

    output_mask_image = np.zeros_like(image)
    for i in range(3):
        output_mask_image[:, :, i] = predict
    output_mask_image = np.where(output_mask_image, image, 0)
    output_segmentation_image = np.where(output_mask_image, (0, 255, 0), image)
    return output_mask_image, output_segmentation_image
