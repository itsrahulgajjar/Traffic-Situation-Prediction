from flask import Flask, render_template, request
import sqlite3
import xgboost as xgb
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError
import time
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Loading the xgboost model
model = xgb.Booster(model_file='xgboost_model-week-12.json')

# SQLite database setup
DB_NAME = 'user_data.db'

# Creating a table to store user data
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        field_1 INTEGER,
        field_2 INTEGER,
        field_3 INTEGER,
        field_4 INTEGER,
        field_5 INTEGER,
        field_6 INTEGER,
        field_7 INTEGER,
        field_8 INTEGER,
        field_9 INTEGER,
        field_10 INTEGER,
        prediction INTEGER
    )
''')
conn.commit()
conn.close()

# Mapping class numbers to traffic situations
class_to_situation = {
    0: 'Low Traffic. You can reach on time wherever you want to go.',
    1: 'Normal Traffic. You can go.',
    2: 'Heavy Traffic. If there is no emergency, you can avoid going out.',
    3: 'High Traffic. If there is no emergency, you can avoid going out.'
}

# AWS S3 configuration
AWS_ACCESS_KEY = 'AKIA47TNT4PF4SUNAEY3'
AWS_SECRET_KEY = '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'
S3_BUCKET = 'week-12-user-data'
S3_REGION = 'us-east-1'

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

@app.route('/')
def index():
    return render_template('index.html')

def visualize_data(user_data):
    # user_data is a list of integers
    labels = ['Day', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Hour', 'Minute', 'AM/PM']

    # Create a bar plot for the distribution of user data values
    plt.figure(figsize=(10, 5))
    plt.bar(labels, user_data[1:9], color='blue')
    plt.title('Distribution of User Data Values')
    plt.xlabel('Fields')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ha='right')

    # Save the first plot to the static folder
    image_path_1 = 'static/user_data_visualization_1.png'
    plt.savefig(image_path_1, format='png')
    plt.close()

    # Create a second plot
    vehicle_types = ['Car', 'Bike', 'Bus', 'Truck']
    counts = user_data[2:6]

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=vehicle_types, autopct='%1.1f%%', startangle=140, colors=['red', 'green', 'blue', 'orange'])
    plt.title('Proportion of Vehicle Types')

    # Save the second plot to the static folder
    image_path_2 = 'static/user_data_visualization_2.png'
    plt.savefig(image_path_2, format='png')
    plt.close()

    return image_path_1, image_path_2

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input data from the form and convert to int
        user_data = [int(request.form[f'field_{i+1}']) for i in range(10)]

        # Perform prediction using the model
        dmatrix = xgb.DMatrix([user_data])
        probabilities = model.predict(dmatrix)

        # Get the predicted class (index with the maximum probability)
        predicted_class = int(probabilities.argmax())

        # Save user input and prediction to the database
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_data (
                field_1, field_2, field_3, field_4, field_5,
                field_6, field_7, field_8, field_9, field_10, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(user_data + [predicted_class]))
        conn.commit()
        conn.close()

        # Upload user data to S3
        upload_data_to_s3(user_data, predicted_class)

        # Visualize user data and save the plot as base64 encoded image
        image_base64_1, image_base64_2 = visualize_data(user_data)

        # Get the corresponding traffic situation
        traffic_situation = class_to_situation.get(predicted_class, 'Unknown')

        return render_template('result.html', prediction=traffic_situation, image_base64_1=image_base64_1, image_base64_2=image_base64_2)
    
def upload_data_to_s3(user_data, predicted_class):
    # Create a unique filename for each user data
    filename = f"user_data_{predicted_class}_{int(time.time())}.txt"

    # Convert user data to a string
    user_data_str = ','.join(map(str, user_data))
    
    try:
        # Upload the user data to S3
        s3.put_object(Bucket=S3_BUCKET, Key=filename, Body=user_data_str)
        print(f"User data uploaded to S3: {filename}")

    except NoCredentialsError:
        print("Credentials not available")

if __name__ == '__main__':
    app.run(debug=True)