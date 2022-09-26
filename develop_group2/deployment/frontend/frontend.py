import streamlit as st
from PIL import Image,ImageOps
import requests
import tensorflow as tf
from io import BytesIO


### Set page 
st.set_page_config(
    page_title="Group 2 : Forest Fire",
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.google.com',
        'Report a bug': "https://github.com/Khal0000",
        'About': "# This is our first dashboard!!!"
    }
)

st.markdown("<h1 style='text-align: center; color:  #ff957f ;'>Forest Fire : Image Segmentation</h1>", unsafe_allow_html=True)
st.markdown("""<hr style="height:10px;border:none;color:#ff957f;background-color:#333;" /> """, unsafe_allow_html=True) 

col1, col2 = st.columns([0.5, 3])

# label = 'label'
with col1:
    source = st.select_slider("source :",options=['Url', 'local'], value='local')

with col2:
    if source == 'local':
        uploaded_file = st.file_uploader("Choose any forest photo ...", type="jpg")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image.', use_column_width=True)
            size = (480, 640)
            image = ImageOps.fit(image, size)
            x = tf.keras.preprocessing.image.img_to_array(image)
            x = x.tolist()

            URL = "https://h8-balls-backend.herokuapp.com/predict"
            data = {"image" : x ,
                "label" : 'M'}

            r = requests.post(URL, json=data)
            res = r.json()

            submit = st.button("Predict")
            if res['code'] == 200 and submit:
                st.markdown(f"<h6 style='text-align: left; color: white ; font-size: 130;'>Our Machine predicted that this is :</h6>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: #34eb95; font-size: 50px;'>{res['result']['classes']}</h5>", unsafe_allow_html=True)
    else :
        url_input = st.text_input('copy and past the url here...', value="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYfcZ0RL2G_rXTaIGZZOy9_Jea0dnHqF2iww&usqp=CAU")
        
        if url_input is not None:
            response = requests.get(str(url_input))
            img = Image.open(BytesIO(response.content))
            st.image(img, caption='Uploaded image.', use_column_width=True)

            size = (460, 640)
            image = ImageOps.fit(img, size)
            x = tf.keras.preprocessing.image.img_to_array(image)
            x = x.tolist()

            URL = "https://h8-balls-backend.herokuapp.com/predict"
            data = {"image" : x ,
                "label" : 'M'}

            r = requests.post(URL, json=data)
            res = r.json()

            submit = st.button("Predict")
            if res['code'] == 200 and submit:
                st.markdown(f"<h6 style='text-align: left; color: white ; font-size: 130;'>Our Machine predicted that this is :</h6>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: #34eb95; font-size: 50px;'>{res['result']['classes']}</h5>", unsafe_allow_html=True)

st.markdown("""<hr style="height:10px;border:none;color:#ff957f;background-color:#333;" /> """, unsafe_allow_html=True) 

