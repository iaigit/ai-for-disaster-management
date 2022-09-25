import streamlit as st
import torch
import torchvision
import models.detection.mobilenetv3 as m # run from repository root as module (python -m )
from PIL import Image
import cv2
import os
from datetime import datetime

st.markdown("<h1 style='text-align: center;'>Forest Fire Detection Application</h1>",
            unsafe_allow_html=True)

st.markdown("<h4>🖼  Detect From Image</h4>", unsafe_allow_html=True)

img = st.file_uploader("Choose a file")
col1, col2 = st.columns(2)
if img is not None:
    img_trans = Image.open(img)
    predict = m.Predict()
    pred, pred_prob = predict.predict(img_trans)
    with col1:
        st.image(img)
    with col2:
        st.markdown(
            f"<h4> Recognition : {pred}</h4>  ", unsafe_allow_html=True)
        st.markdown(
            f"<h4> Recognition Probs : {pred_prob}</h4>  ", unsafe_allow_html=True)


st.markdown("<h4> 📸 Detect From Videos</h4>", unsafe_allow_html=True)
vid = st.file_uploader("Choose a file", key="vids")
if vid is not None:
    ts = datetime.timestamp(datetime.now())
    vidpath = os.path.join('data/uploads', str(ts)+vid.name)
    outputpath = os.path.join(
        'data/outputs', os.path.basename(vidpath))
    with open(vidpath, mode="wb") as f:
        f.write(vid.getbuffer())
    vid = cv2.VideoCapture(vidpath)
    vid.set(cv2.CAP_PROP_FPS, 1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 254)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 254)
    vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'))

    predict = m.Predict()
    while vid.isOpened():
        ok, frame = vid.read()
        if not ok:
            break
        pred, pred_prob = predict.predict(frame)
        cv2.putText(frame, pred, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(pred_prob), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )


# st.write("Refresh halaman atau ulangi klik tombol start jika camera tidak muncul")
# font = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (20, 40)
# fontScale = 1
# fontColor = (50, 168, 82)
# thickness = 2
# lineType = 1


# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#     img_trans = Image.fromarray(img)
#     pred, pred_prob = m.Predict().predict(img_trans)
#     if pred == "with mask":
#         fontColor = (12, 92, 7)
#     elif pred == "without mask":
#         fontColor = (247, 42, 35)
#     cv2.putText(img, pred + " " + str(pred_prob),
#                 bottomLeftCornerOfText,
#                 font,
#                 fontScale,
#                 fontColor,
#                 thickness,
#                 lineType)

#     return av.VideoFrame.from_ndarray(img, format="bgr24")
