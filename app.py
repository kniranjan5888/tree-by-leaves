from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2
import os
import numpy as np
from flask_cors import CORS


# recognizer = cv2.face.LBPHFaceRecognizer_create()


app = Flask(__name__)
CORS(app)

dic = {
    0: "Acer",
    1: "Alnus incana",
    2: "Betula pubescens",
    3: "Fagus silvatica",
    4: "Populus",
    5: "Populus tremula",
    6: "Quercus",
    7: "Salix alba 'Sericea'",
    8: "Salix aurita",
    9: "Salix sinerea",
    10: "Sorbus aucuparia",
    11: "Sorbus intermedia",
    12: "Tilia",
    13: "Ulmus carpinifolia",
    14: "Ulmus glabra",
}

path = os.path.join(
    os.path.dirname(__file__), "VGG_16_model_Augmented_best_weights_model.h5"
)
print(path)
model = load_model(path)


def gray2img(folder):
    images = np.ndarray(
        shape=(folder.shape[0], folder.shape[1], folder.shape[2], 3), dtype=np.uint8
    )
    images[:, :, :, 0] = folder[:, :, :, 0]
    images[:, :, :, 1] = folder[:, :, :, 0]
    images[:, :, :, 2] = folder[:, :, :, 0]
    return images


def predict_label(img_path, model):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 1)
    img = gray2img(img)
    p = model.predict(img)
    p = [float(i) * 100 for i in p[0]]
    d = dic.values()
    pred = dict(zip(d, p))
    return pred


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def upload():
    if request.method == "POST":
        input_file = request.files["file"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", input_file.filename)
        input_file.save(file_path)
        preds = predict_label(file_path, model)
        return preds
    return None


if __name__ == "__main__":
    app.run(debug=True, port=5000)
