from flask import Flask, request, render_template
from flask import Response
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import pytesseract
from preprocess import Preprocess
import pandas as pd
from google.cloud import vision
import io
import warnings
import pyttsx3
# engine = pyttsx3.init()

warnings.simplefilter("ignore")

# This is the project key which needs for using vision API
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = "F:\Projects\Handwritten Digits Recognition\GoogleKey\OCRKey_Extc.json"

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    try:
        if os.path.exists('Uploads/img.jpeg'):
            os.remove("Uploads/img.jpeg")

        if request.method == 'POST':
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join('Uploads', filename))
                os.rename(os.path.join('Uploads', filename), os.path.join('Uploads', "img.jpeg"))
            return Response("Upload successfull!!")
    except Exception as e:
        print(e)
        raise Exception()


def extract_using_pytesseract(image_roi):
    return pytesseract.image_to_string(image_roi)


def CloudVisionTextExtractor(handwritings):
    # convert image from numpy to bytes for submittion to Google Cloud Vision
    _, encoded_image = cv2.imencode('.png', handwritings)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)

    # feed handwriting image segment to the Google Cloud Vision API
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image)

    return response


def getTextFromVisionResponse(response):
    texts = []
    for page in response.full_text_annotation.pages:
        for i, block in enumerate(page.blocks):
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    texts.append(word_text)

    return ' '.join(texts)


def text_to_speech(text, gender):

    voice_dict = {'Male': 0, 'Female': 1}
    code = voice_dict[gender]

    engine = pyttsx3.init()

    # Setting up voice rate
    engine.setProperty('rate', 125)

    # Setting up volume level  between 0 and 1
    engine.setProperty('volume', 0.8)

    # Change voices: 0 for male and 1 for female
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[code].id)

    engine.say(text)
    engine.runAndWait()


@app.route("/extract", methods=['POST'])
@cross_origin()
def extract_text():
    try:
        path = "Uploads/img.jpeg"

        image = cv2.imread(path, 0)
        prep = Preprocess(image)
        thresholded_image = prep.threshold_image()
        largest_contours, image_with_largest_contours = prep.findAndDrawContour(thresholded_image)
        text_contour, image_with_text_contour = prep.image_with_text_contour(largest_contours,
                                                                             image_with_largest_contours)
        image_roi = prep.warp_perspective(image, text_contour)
        prep = Preprocess(image_roi)
        thresholded_image = prep.threshold_image()
        dilated = prep.morphological_dilation(image_roi, 10)
        prep = Preprocess(dilated)
        thresholded_image = prep.threshold_image(adaptive=True)
        closed = prep.morphological_closing(thresholded_image)
        median = cv2.medianBlur(closed, 3)
        erosion_1 = prep.morphological_erosion(median, 3)
        compare_1 = np.concatenate((image_roi, erosion_1), axis=1)  # side by side comparison
        text = extract_using_pytesseract(image_roi)
        print(text)
        response = CloudVisionTextExtractor(image_roi)
        handwrittenText = getTextFromVisionResponse(response)
        print(handwrittenText)
        # engine.say(handwrittenText)
        # engine.runAndWait()
        text_to_speech(handwrittenText, 'Male')
        return Response(handwrittenText)

    except ValueError:
        return Response("Value Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("KeyError Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Exception Error Occurred! %s" % e)


if __name__ == "__main__":
    app.run(debug=False)
