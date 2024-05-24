from flask import Flask, request, jsonify, render_template
from gradio_client import Client, file
from PIL import Image
import numpy as np
import io
import os, shutil

# Import your image classification model
# Example:
# from my_image_classifier import ImageClassifier
# classifier = ImageClassifier()

app = Flask(__name__)

# connecting to a Hugging Face Space
client = Client("Pratik-hf/inappropriate-content-detection")

def classify_text(text):
    result = client.predict(
		text,
		api_name="/classify_text"
    )
    predicted_class = result[-1]
    return predicted_class


def classify_image(image):
    result = client.predict(
		file(image),
		api_name="/classify_image"
    )
    predicted_class = result[-1]
    return predicted_class


@app.route('/classify-text', methods=['POST', 'GET'])
def classify_text_view():
    if request.method == 'POST':
        try:
            text = request.form['text']
            if text:
                result = classify_text(text)

                if result[0] == 0:
                    result = "Appropriate"
                else:
                    result = "Inappropriate"
            else:
                result = 'Please Provide valid input'
        except Exception as e:
            result = 'Something went wrong!'

        return jsonify({"response": result})
    else:
        return render_template("classify-text.html", data={})


@app.route('/classify-image', methods=['POST', 'GET'])
def classify_image_view():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})

        image_file = request.files['image']

        upload_dir = 'uploads'

        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        image_path = os.path.join(upload_dir, image_file.filename)
        image_file.save(image_path)

        # Classify the image
        prediction = classify_image(image_path)

        return jsonify({"response": prediction})

    else:
        return render_template("classify-image.html", data={})

@app.route('/')
def index_view():
    return render_template('index.html')