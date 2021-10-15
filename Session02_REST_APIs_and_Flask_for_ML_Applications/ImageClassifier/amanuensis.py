from flask import Flask, render_template, request, flash, redirect, url_for
from flask import get_flashed_messages

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from torch import nn
import torch.nn.functional as F

import os
import io
import base64
import json

from model import *

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]]]]/'

# set device to cpu
device = torch.device('cpu')
NUM_CLASSES = 10
CIFAR10_CLASS_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model = ResNet18(num_classes=NUM_CLASSES).to(device)

PATH = "cifar10_aug_model_l1_l2.pth"
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
model.eval()


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))
        ])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_tensor):
    try:
        images = image_tensor
        outputs = model(images)
        conf_list = torch.nn.functional.softmax(outputs.data[0],dim=0).numpy().tolist()
        #print(conf_list)
        _, predicted = torch.max(outputs.data, 1)
        
        return predicted
    except Exception as e:
        raise(e)

@app.route('/')
def index():
    get_flashed_messages()
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            image_file = request.files.get('imageupload')
            if image_file is not None and image_file.filename != '':
                image = image_file.read()
                img = base64.b64encode(image).decode("utf-8")
                image_url = 'data:image/*;base64,{}'.format(img)
                if not allowed_file(image_file.filename):
                    flash("Image format not supported. Allowed: png,jpg,jpeg")
                    return render_template('index.html',  image=img,  set_tab=1)
                else:
                    tensor = transform_image(image)
                    prediction = get_prediction(tensor)
            else:
                if image_file is None or image_file.filename == "":
                    flash("Please select an image")
                
                #return redirect(url_for('index', _anchor='services'))
                return render_template('index.html',  set_tab=1)
        return render_template('index.html',  image=img, caption=str(CIFAR10_CLASS_LABELS[int(prediction.item())]).upper(), set_tab=1)
    except Exception as e:
        flash("Please select valid image of png,jpg,jpeg format")
        return render_template('index.html', set_tab=1)

if __name__ == '__main__':
    app.run(host='0.0.0.0')#, debug=True)
