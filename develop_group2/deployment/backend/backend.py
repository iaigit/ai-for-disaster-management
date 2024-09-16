from flask import Flask, jsonify, request
import pandas as pd
import tensorflow as tf
import numpy as np
# from tensorflow.keras.preprocessing import image

app = Flask(__name__)

CLASS = ['baseball', 'basketball','beachballs','billiard ball','bowling ball',
        'brass','buckeyballs','cannon ball','cricket ball','eyeballs',
        'football','golf ball','marble','meat ball','medicine ball',
        'paint balls','pokeman balls','puffballs','screwballs','soccer ball',
        'tennis ball','tether ball','volley ball','water polo ball','wiffle ball', 'wrecking ball']

columns = ["image", 'label']


# import model

my_model = tf.keras.models.load_model('model_best.hdf5')

@app.route("/")
def hello_world():
    return "<p>Hello, This is my Backend Data!</p>"


@app.route("/predict", methods=['GET', 'POST'])
def body_inference():
    if request.method == 'POST':
        data = request.json
        new_data =[data["image"],
                    data["label"]]

        new_data = pd.DataFrame([new_data], columns = columns)

        # Data Augmentation
        path = new_data["image"][0]
        # img = tf.keras.preprocessing.image.load_img(path[0], target_size=(img_height, img_width))
        # x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(path, axis=0)
        x = x * 1./255

        images = np.vstack([x])

        predict0 = my_model.predict(images)
        res = np.argmax(predict0, axis=1)

        response = {'code':200, 'status':'OK',
                    'result':{'prediction': str(res[0]),
                    'classes': CLASS[res[0].item()]
                    }}
        
        return jsonify(response)
    return "Connection succesfully established"

# app.run(debug=True)
