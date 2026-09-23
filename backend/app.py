# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_predictor_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model pipeline
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_sales_predictor_api.get('/')
def home():
    """
    Handles GET requests to the root URL ('/').
    Returns a simple status welcome message.
    """
    return "Welcome to the SuperKart Product Store Sales Prediction API!"

# Define an endpoint for single product/store prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predict')
def predict_sales():
    """
    Handles POST requests to the '/v1/predict' endpoint.
    Expects a JSON payload containing product and store details,
    applies feature transformations, and returns predicted total sales.
    """
    try:
        # Get JSON payload from request body
        data = request.get_json()

        # Extract relevant features from payload
        sample = {
            'Product_Weight': float(data['Product_Weight']),
            'Product_Sugar_Content': data['Product_Sugar_Content'],
            'Product_Allocated_Area': float(data['Product_Allocated_Area']),
            'Product_Type': data['Product_Type'],
            'Product_MRP': float(data['Product_MRP']),
            'Store_Size': data['Store_Size'],
            'Store_Location_City_Type': data['Store_Location_City_Type'],
            'Store_Type': data['Store_Type'],
            'Store_Age': int(data['Store_Age']),
            'Product_Category_Code': data['Product_Category_Code']
        }

        # Convert dictionary into a single-row Pandas DataFrame
        input_data = pd.DataFrame([sample])

        # Generate prediction using the serialized pipeline
        predicted_sales = model.predict(input_data)[0]

        # Convert predicted value to Python float and round
        predicted_sales = round(float(predicted_sales), 2)

        # Return prediction as JSON response
        return jsonify({
            'status': 'success',
            'Predicted Store Sales Total ($)': predicted_sales
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define an endpoint for batch prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predict_batch')
def predict_sales_batch():
    """
    Handles POST requests to the '/v1/predict_batch' endpoint.
    Expects a CSV file containing multiple product/store entries
    and returns predicted sales mapped to unique Product_Id / Store_Id pairs.
    """
    try:
        # Get uploaded CSV file from request
        file = request.files['file']

        # Read CSV file into a Pandas DataFrame
        input_data = pd.read_csv(file)

        # Ensure feature engineering matches preprocessing pipeline requirements
        if 'Store_Establishment_Year' in input_data.columns and 'Store_Age' not in input_data.columns:
            input_data['Store_Age'] = 2026 - input_data['Store_Establishment_Year']

        if 'Product_Id' in input_data.columns and 'Product_Category_Code' not in input_data.columns:
            input_data['Product_Category_Code'] = input_data['Product_Id'].str[:2]

        if 'Product_Sugar_Content' in input_data.columns:
            input_data['Product_Sugar_Content'] = input_data['Product_Sugar_Content'].replace({'reg': 'Regular'})

        # Make predictions across all rows
        predicted_sales = model.predict(input_data).tolist()

        # Format predicted values
        formatted_predictions = [round(float(sales), 2) for sales in predicted_sales]

        # If Identifier columns exist, create a mapped response dictionary
        if 'Product_Id' in input_data.columns and 'Store_Id' in input_data.columns:
            identifiers = (input_data['Product_Id'] + "_" + input_data['Store_Id']).tolist()
            output_dict = dict(zip(identifiers, formatted_predictions))
            return jsonify({'status': 'success', 'predictions': output_dict}), 200
        else:
            return jsonify({'status': 'success', 'predictions': formatted_predictions}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run Flask application on port 5000
if __name__ == '__main__':
    superkart_sales_predictor_api.run(host="0.0.0.0", port=5000, debug=True)
