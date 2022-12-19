import os
import numpy as np

from utils import predict_caption, text_to_audio
from flask import Flask,render_template, request


app = Flask(__name__,static_url_path='',static_folder='temp', template_folder='templates')

@app.route("/")
def index():
    return render_template('index.html')

    
@app.route("/predict", methods = ["POST"])
def predict():
    img_file = request.files['file']
    img_name = img_file.filename
    img_path = os.path.join("./temp/imgs/", img_name)
    img_file.save(img_path)
    pred_caption = predict_caption(img_path)
    print(img_path,'img_path')
    print(pred_caption, 'pred_caption')
    audio_name = text_to_audio(pred_caption)
    audio_path = os.path.join("./temp/audios/", audio_name)
    print(audio_path,'audio_path')
    return render_template('predict.html', pred_caption=pred_caption, img_path=f"/imgs/{img_name}", audio_path=f"/audios/{audio_name}.mp3")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    
    