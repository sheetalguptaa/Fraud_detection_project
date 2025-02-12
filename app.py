import pickle
from flask import Flask, render_template, request
import pandas as pd
import pymongo
from pymongo import MongoClient

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and LabelEncoders from the .pkl files
with open('random_forest_model.pkl', 'rb') as model_file:
    rf_classifier = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Define categorical columns used for training
categorical_cols = ['merchant', 'category', 'gender', 'job']

# Connect to MongoDB Atlas
cluster = MongoClient("mongodb+srv://sheetal51:12345@cluster0.fg2ic.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0") 
db = cluster["fraud_detection"]
collection = db["user_inputs"]

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract user input from form submission
    user_data = {
        'merchant': request.form['merchant'],
        'category': request.form['category'],
        'amt': float(request.form['amt']),
        'gender': request.form['gender'],
        'job': request.form['job'],
        'lat': float(request.form['lat']),
        'long': float(request.form['long']),
        'city_pop': float(request.form['city_pop'])
    }

    # Convert user input to DataFrame correctly (ensure proper 2D structure)
    input_data = pd.DataFrame([user_data])

    # Transform categorical columns using stored LabelEncoders
    for col in categorical_cols:
        if user_data[col] in label_encoders[col].classes_:
            input_data[col] = label_encoders[col].transform([user_data[col]])
        else:
            input_data[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])

    # Make prediction using the trained model
    prediction = rf_classifier.predict(input_data)[0]  # Extract single prediction

    # Store user input + prediction result in MongoDB
    user_data["prediction"] = str(prediction)  # Convert NumPy type to string
    collection.insert_one(user_data)

    # Display prediction result on frontend
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
