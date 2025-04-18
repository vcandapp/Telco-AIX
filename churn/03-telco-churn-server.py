# Author: Ömer Saatcioglu

import os
import threading
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pickle

def process_data(data):
    # Mapping for PaymentStatus
    payment_mapping = {"Paid": 0, "Partial": 0.5, "Unpaid": 1}

    # Mapping for PrimaryIssueType: empty string indicates no support interaction.
    issue_mapping = {
        "": 0, 
        "Billing": 0.7, 
        "Technical": 0.8, 
        "Service Quality": 0.6
    }

    # Mapping for SupportChannel: adjust the numeric values as needed.
    support_channel_mapping = {
        "": 0,       # No support channel if there's no interaction.
        "Phone": 1,  # Example value: Phone might be considered more direct.
        "Chat": 0.5, # Example value: Chat might be intermediate.
        "Email": 0.2 # Example value: Email might be considered less immediate.
    }

    # Convert to numeric values
    data["PaymentStatus"] = data["PaymentStatus"].map(payment_mapping)
    data["PrimaryIssueType"] = data["PrimaryIssueType"].map(issue_mapping)
    data["SupportChannel"] = data["SupportChannel"].map(support_channel_mapping)

    # The date columns
    date_columns = ['BillingCycleStart', 'BillingCycleEnd', 'PaymentDueDate', 'LastInteractionDate']

    # drop date columns and customer ID
    data = data.drop(columns=['CustomerID'] + date_columns, axis=1)

    return data


load_dotenv(override=True)

base_model_brfc = os.getenv("MODEL_CHURN_BRFC", "models/customer_churn_prediction_model_brfc.pkl")
base_model_lgbm = os.getenv("MODEL_CHURN_LGBM", "models/customer_churn_prediction_model_lgbm.pkl")

with open(base_model_brfc, 'rb') as model_file:
    model_brfc, feature_names_brfc = pickle.load(model_file)

with open(base_model_lgbm, 'rb') as model_file:
    model_lgbm, feature_names_lgbm = pickle.load(model_file)

# Initialize Flask application
app = Flask(__name__)

# Define a route for the default URL, which serves a simple welcome message
@app.route('/')
def home():
    return "Welcome to the Churn Prediction Model Server!"

# Define a route for the prediction API with balanced random forest classifier
@app.route('/predict-brfc', methods=['POST'])
def predict_brfc():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Convert the JSON data to a DataFrame
    input_data = pd.DataFrame([data])

    # Process the input data
    input_data = process_data(input_data)

    # Ensure the columns match the training data
    input_data = input_data.reindex(columns=feature_names_brfc, fill_value=0)

    # Make predictions using the loaded model
    prediction = model_brfc.predict(input_data)
    
    # Map the prediction to a more user-friendly response
    prediction_label = 'Will Churn' if prediction[0] == 1 else 'Will Not Churn'
    
    # Return the prediction result as JSON
    return jsonify({'Prediction Result': f': {prediction_label}'})

# Define a route for the prediction API with LightGBM classifier
@app.route('/predict-lgbm', methods=['POST'])
def predict_lgbm():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Convert the JSON data to a DataFrame
    input_data = pd.DataFrame([data])
    
    # Process the input data
    input_data = process_data(input_data)

    # Ensure the columns match the training data
    input_data = input_data.reindex(columns=feature_names_lgbm, fill_value=0)

    # Make predictions using the loaded model
    prediction = model_lgbm.predict(input_data)
    
    # Map the prediction to a more user-friendly response
    prediction_label = 'Will Churn' if prediction[0] == 1 else 'Will Not Churn'
    
    # Return the prediction result as JSON
    return jsonify({'Prediction Result': f': {prediction_label}'})

# Function to run the Flask app in a separate thread
def run_app():
    app.run(debug=False, host='0.0.0.0', port=35000)

# Start the Flask app in a separate thread
if __name__ == '__main__':
    threading.Thread(target=run_app).start()
