import base64
import json
import os
import numpy as np
import requests

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

from architecture import LeNet, preprocess, predict_top_1, predict_top_3

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


app = Flask(__name__)


def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'weights.h5')
    model = LeNet()
    model.load_weights(model_path)
    return model


def read_image(files):
    image = Image.open(files)
    image = np.array(image)
    return image

@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, world'


@app.route('/image/', methods=['POST'])
def image():
    # Byte file
    files = request.files['image']
    # img = Image.frombytes(BytesIO(base64.b64decode(files)))
    image = Image.open(files)
    image = np.array(image)

    return {"image_shape": image.shape}


@app.route('/predict/',methods=['GET','POST'])
def predict():
    model = load_model()
    files = request.files['image']
    image = read_image(files)
    image = preprocess(image)
    # response = np.array_str(np.argmax(out,axis=1))
    predicted = model(image)

    top_1 = predict_top_1(predicted)
    top_3 = predict_top_3(predicted)

    print(top_3)
    response = np.array_str(top_1.numpy())
    # return {"status":"ok"}	
    return response


if __name__ == "__main__":
    app.run()