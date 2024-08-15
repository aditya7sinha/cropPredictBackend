from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from flask_cors import CORS
import joblib
from config import Config


# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Load the models
tf_model = tf.keras.models.load_model('crop_model.h5')  # TensorFlow model
rf_model = joblib.load('random_forest_model.joblib')  # RandomForest model
# Load the label encoder
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/rainfall2', methods=['POST'])
def predict_rainfall2():
    return "works"

# Route for predicting rainfall using the RandomForest model
@app.route('/rainfall', methods=['POST'])
def predict_rainfall():
    data = request.json
    district = data['district']
    year = data['year']
    month = data['month']
    
    # Encode the district and make the prediction
    try:
        district_encoded = label_encoder.transform([district])[0]
        features = [[district_encoded, year, month]]
        prediction = rf_model.predict(features)
        return jsonify({'predicted_rainfall': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route for predicting crop using the TensorFlow model
@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    # Get the JSON data from the request
    data = request.json
    
    # Convert incoming data into a DataFrame
    input_data = pd.DataFrame([data])
    
    # Make prediction using the TensorFlow model
    try:
        prediction = tf_model.predict(input_data)
        predicted_label = np.argmax(prediction, axis=1)
        
        # Map prediction to crop name
        crop_mapping = [
            "Rice", "Maize", "Jute", "Cotton", "Coconut", "Papaya", "Orange", "Apple",
            "Muskmelon", "Watermelon", "Grapes", "Mango", "Banana", "Pomegranate", "Lentil",
            "Blackgram", "MungBean", "MothBeans", "PigeonPeas", "KidneyBeans", "ChickPea", "Coffee"
        ]
        predicted_crop = crop_mapping[predicted_label[0]]
        return jsonify({'predicted_crop': predicted_crop})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
