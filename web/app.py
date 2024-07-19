import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2

import tensorflow as tf

TestResult = ["Possitive","Negative"]

symptom_model = tf.keras.models.load_model(r'Model\symptomsModel.h5')
conjunctiva_model = tf.keras.models.load_model(r'Model\conjunctiva_100.h5')
fingerNail_model = tf.keras.models.load_model(r'Model\fingerNail_100.h5')
palm_model = tf.keras.models.load_model(r'Model\palm_100.h5')





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

def predict_anemia(model, img_path, image_size=(128, 128)) -> float:
    # Load and preprocess the image
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)

        # Convert the prediction to a binary class label
        threshold = 0.5
        class_label = (prediction > threshold).astype(int)

        print(prediction[0][0])
        return prediction[0][0]
        # # Output the result
        # if class_label == 0:
        #     return "Not Anemic"
        # else:
        #     return "Anemic"
    else:
        return 0

def predict_conjunctiva():
    return float(predict_anemia(conjunctiva_model, r'web\uploads\uploaded_image1.png'))


def predict_palm(): 
    return float(predict_anemia(palm_model, r'web\uploads\uploaded_image2.png'))


def predict_fingerNail(): 
    return float(predict_anemia(fingerNail_model, r'web\uploads\uploaded_image3.png'))



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
            image_path = os.path.join('web', 'uploads', 'uploaded_image1.png')
            image = Image.open(image_file)
            image.save(image_path)
        else:
            return jsonify({'error': 'Image1 file not found'}), 400
        
        image_file = request.files['image2']
        if image_file.filename != '':
            # Save the uploaded image to a temporary folder
            image_path = os.path.join('web', 'uploads', 'uploaded_image2.png')
            image = Image.open(image_file)
            image.save(image_path)
        else:
            return jsonify({'error': 'Image2 file not found'}), 400
        
        image_file = request.files['image3']
        if image_file.filename != '':
            # Save the uploaded image to a temporary folder
            image_path = os.path.join('web', 'uploads', 'uploaded_image3.png')
            image = Image.open(image_file)
            image.save(image_path)
        else:
            return jsonify({'error': 'Image3 file not found'}), 400

    else:
        return jsonify({'error': 'Image file not found'}), 400

    # Perform prediction using the trained model
    symptom_result = (predict_symptoms(symptoms) + 0.46) * 100.0
    conjunctiva_result = (predict_conjunctiva() *100.0)
    fingerNail_result = (predict_fingerNail() *100.0)
    palm_result = (predict_palm()*100.0)

    # Return the prediction as a JSON response
    avgScore = (symptom_result + conjunctiva_result + fingerNail_result + palm_result ) / 4

    if avgScore > 50:
        result = 0
    else:
        result = 1

    return jsonify({
        "result": f"{TestResult[result]}",
        "symptom_res": f"{symptom_result:.2f}%",
        "conjunctiva_res": f"{conjunctiva_result:.2f}%",
        "fingernail_res": f"{fingerNail_result:.2f}%",
        "palm_res": f"{palm_result:.2f}%"
    })


@app.route("/moreinfo")
def moreinfo():
    return render_template("moreinfo.html")

if __name__ == '__main__':
    app.run("0.0.0.0",debug=True)
