from flask import Flask, render_template, request, redirect, session
import os
import sys
import pickle5
import numpy as np
from PIL import Image
from transforms import fishTransform
from pathlib import Path

app = Flask(__name__)

class SoftMax(object):
    def __init__(self):
        pass

    def __call__(self, input):
        return np.exp(input) / np.exp(input).sum()

class SaveImage(object):
    def __init__(self, image_file):
        save_path = "./image/image.jpg"
        self.save_path = Path(save_path)
        img = Image.open(image_file)
        img.save(save_path)

class MlModel(object):
    def __init__(self, model_type):
        # choose model
        if model_type == "nokoshima":
            self.weight_path = "./models/nokoshima/latest.pkl"
            self.num_classes = 2
            self.class_names = ["ノコギリハギ", "シマキンチャクフグ"]
    
    def __call__(self, image_file):
        # load model
        with open(self.weight_path, "rb") as f:
           model = pickle5.load(f)
        
        model.eval()

        # input image
        #print(image_file)
        #save_image = SaveImage(image_file)
        #img = Image.open(save_image.save_path)
        img = Image.open(image_file)

        # transfrom
        transform = fishTransform(resize=(256,256), mean=None, std=None)
        transformed_img = transform(img, key='val')
        transformed_img = transformed_img.to('cpu')
        input = transformed_img.unsqueeze(0)

        # prediction
        m = SoftMax()
        output = model(input)
        output = output.detach().numpy()
        output = m(output)[0]
        r = np.argmax(output)

        result = self.class_names[r]
        pred = output[r].item()

        #print("予測結果：", result)
        #print("予測確率：{:.3f}".format(pred))

        return result, round(pred, 3)

@app.route("/", methods=["GET", "POST"])
def start():
    if request.method == "POST":
        model_type = request.form.get("model_type")
        if model_type == "nokoshima":
          return redirect('/nokoshima_clf')
    else:
        return render_template("start.html")

@app.route("/detail", methods=["GET", "POST"])
def detail():
        return render_template("detail.html")


@app.route("/nokoshima_clf", methods=["GET", "POST"])
def nokoshima_clf():
    if request.method == "POST":

        image_file = request.files["image_file"]
        model = MlModel(model_type="nokoshima")
        #print("img: ", image_file)
        #print("model: ", model)
        result, pred= model(image_file=image_file)

        if os.path.isfile("image/image.jpg"):
            os.remove("image/image.jpg")

        return render_template("nokoshima_clf.html", result=result, pred=pred)

    else:
        return render_template("nokoshima_clf.html", result=None, pred=None)

if __name__ == "__main__":
    app.run(debug=False)