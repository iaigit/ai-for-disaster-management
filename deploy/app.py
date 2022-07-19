import cv2
import time
import tempfile
import streamlit as st
import onnxruntime as ort
from PIL import Image
from helpers import predict_image

DEVICE_INFERENCE = "CPU"
MODEL_PATH = "../experiment-active_learning/all_datasets/onnx_inference_model.onnx"

# Set page config
st.set_page_config(
    page_title="Indonesia AI - RnD Team for Disaster Management",
    page_icon="🔥",
)

# Load model
@st.cache(allow_output_mutation=True)
def load_model(model_path, device_inference="cpu"):
    if device_inference.lower() == "cpu":
        ort_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    elif device_inference.lower() == "cuda":
        ort_session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"],
        )
    else:
        st.error("Please select between CPU or CUDA!")
        st.stop()

    return ort_session


# Run load model
model = load_model(MODEL_PATH, DEVICE_INFERENCE)

# Main page
st.title("Indonesia AI - RnD Team for Disaster Management")
st.write(
    """
        Forest fires are considered as one of the most widespread hazards in a forested landscape. They have a serious threat to forest and its flora and fauna. Forest fires prediction combines weather factors, terrain, dryness of flammable items, types of flammable items, and ignition sources to analyze and predict the combustion risks of flammable items in the forest.
"""
)
st.markdown("  ")

format_file = st.selectbox("Select format file to predict", ["Image", "Video"])
if format_file.lower() == "image":
    uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file = Image.open(uploaded_file)
        st.markdown("  ")
        st.write("Source Image")
        st.image(uploaded_file)

        predict_button = st.button("Detect forest fire")
        st.markdown("  ")

        if predict_button:
            with st.spinner("Wait for it..."):
                start_time = time.time()
                mask_image, segmentation_image = predict_image(uploaded_file, model)
                col1, col2 = st.columns(2)
                col1.write("Mask Image")
                col1.image(mask_image)
                col2.write("Segmentation Image")
                col2.image(segmentation_image)
                st.write(f"Inference time: {(time.time() - start_time):.3f} seconds")

elif format_file.lower() == "video":
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi"])
    if uploaded_file is not None:
        st.markdown("  ")
        st.write("Source Video")
        st.video(uploaded_file)
        predict_button = st.button("Detect forest fire")
        st.markdown("  ")
        if predict_button:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                cap = cv2.VideoCapture(tmp.name)
                col1_text, col2_text = st.columns(2)
                col1_text.write("Mask Image")
                col2_text.write("Segmentation Image")
                col1_image, col2_image = st.columns(2)
                col1_image_empty, col2_image_empty = (
                    col1_image.empty(),
                    col2_image.empty(),
                )
                while cap.isOpened():
                    _, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    mask_image, segmentation_image = predict_image(frame, model)
                    col1_image_empty.image(
                        mask_image,
                    )
                    col2_image_empty.image(
                        segmentation_image,
                    )
                cap.release()
