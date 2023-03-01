import argparse
import streamlit as st
import requests
import pandas as pd
import json
import sqlite3
from PIL import Image

button1 = st.button('Check 1')

if st.session_state.get('button') != True:

    st.session_state['button'] = button1

# Set app title
st.title('Animal Classifier')

# Define a placeholder for the uploaded file data
image = None

# Define a placeholder for the predictions
predictions = None

# Use st.file_uploader to get a file from the user
st.caption("Choose an image for classification here")
file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'], label_visibility='visible')

# If a file was uploaded
if file is not None:

    # Convert image to bytes
    image_bytes = file.read()

    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image')


# If the user has uploaded an image and wants to make predictions
if image is not None and st.session_state['button'] == True:
    print("Button1")
    st.caption("Prediction in progress...")
    # Use st.spinner to display a spinner while predictions are being made
    with st.spinner('Predicting...'):
        # Define the FastAPI endpoint URL
        api_url = 'http://localhost:8000/predict'

        # Make API request
        response = requests.post(api_url, files={'file': image_bytes})

        # Check if request was successful
        if response.status_code == 200:
            # Parse predictions from response
            predictions = response.json()

        # If request failed, raise an exception
        else:
            raise Exception('API request failed with status code {}'.format(response.status_code))
    
    label = predictions["response"]
    # Display the prediction results
    st.subheader(f'That animal is a {label}')

    if st.button('Check 2'):
            print("Clicked")
            st.caption("Recommending images...")
            
            # Connect to the SQLite database
            conn = sqlite3.connect("databases/images.db")

            cursor = conn.cursor()
            # Get a list of 5 random images from the database
            query = f"SELECT file_path FROM metadata WHERE label='{label}' ORDER BY RANDOM() LIMIT 5"
            cursor.execute(query)
            rows = cursor.fetchall()
            images = [row[0] for row in rows]

            # Display the images using Streamlit
            for img_path in images:
                image = Image.open(img_path)
                st.image(image, caption=img_path)

            conn.close()

            st.session_state['button'] = False

# # If predictions have been made, display a download button for the predictions
# if predictions is not None:
#     # Use st.download_button to allow the user to download the predictions as a CSV file
#     st.download_button(
#         label='Download predictions',
#         data=json.dumps(predictions),
#         file_name='predictions.json',
#         mime='application/json'
#     )
