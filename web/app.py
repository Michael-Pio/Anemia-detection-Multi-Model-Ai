import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np

import tensorflow as tf

TestResult = ["Possitive","Negative"]

symptom_model = tf.keras.models.load_model('Model\symptomsModel.h5')

# Preprocess the symptoms to a one-hot encoded array
def preprocess_symptoms(symptoms):
    # Converting symptoms to a one-hot encoded array (0: Absent, 1: Present)
    all_symptoms = ["General Fatigue", "Weakness", "Shortness of Breath", "Pale Skin",
                    "Rapid Heartbeat", "Dizziness or Lightheadedness", "Cold Hands and Feet",
                    "Brittle Nails", "Headaches", "Poor Concentration", "Chest Pain", "Restless Legs"]
    symptoms_array = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    return np.array(symptoms_array).reshape(1, -1)


# Use the trained model to predict the result
def predict_symptoms(symptoms):
    symptoms_array = preprocess_symptoms(symptoms)
    prediction = symptom_model.predict(symptoms_array)
    print("Predicted :",prediction)
    return prediction[0][1] - 0.46

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')


# Process the uploaded image and return the prediction
@app.route('/process', methods=['POST'])
def process():
    # Get symptom data from the form submission
    symptoms = request.form.getlist('symptom')

    # Process the uploaded image
    if (('image1' in request.files) and ('image2'in request.files) and ('image3' in request.files)):
        image_file = request.files['image1']
        if image_file.filename != '':
            # Save the uploaded image to a temporary folder
            image_path = os.path.join('web', 'static', 'uploaded_image1.png')
            image = Image.open(image_file)
            image.save(image_path)
        else:
            return jsonify({'error': 'Image file not found'}), 400
        
        image_file = request.files['image2']
        if image_file.filename != '':
            # Save the uploaded image to a temporary folder
            image_path = os.path.join('web', 'static', 'uploaded_image2.png')
            image = Image.open(image_file)
            image.save(image_path)
        else:
            return jsonify({'error': 'Image file not found'}), 400
        
        image_file = request.files['image3']
        if image_file.filename != '':
            # Save the uploaded image to a temporary folder
            image_path = os.path.join('web', 'static', 'uploaded_image3.png')
            image = Image.open(image_file)
            image.save(image_path)
        else:
            return jsonify({'error': 'Image file not found'}), 400

    else:
        return jsonify({'error': 'Image file not found'}), 400

    # Perform prediction using the trained model
    prediction = predict_symptoms(symptoms)

    if prediction > 0.5:
        result = 0
    else:
        result = 1

    confidence = (prediction + 0.46) * 100

    return jsonify({
        "result": f"{TestResult[result]}",
        "confidence": f"{confidence:.2f}%"
    })


@app.route("/moreinfo")
def moreinfo():
    return render_template("moreinfo.html")

if __name__ == '__main__':
    app.run("0.0.0.0",debug=True)
