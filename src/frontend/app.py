import argparse
import streamlit as st
import requests
import pandas as pd
import json
from PIL import Image
      
# Define function to make API request and return predictions
def get_predictions(file_bytes):

    # Define the FastAPI endpoint URL
    api_url = args.endpoint

    # Make API request
    response = requests.post(api_url, files={'file': file_bytes})

    # Check if request was successful
    if response.status_code == 200:
        # Parse predictions from response
        predictions = response.json()

        # Return predictions
        return predictions

    # If request failed, raise an exception
    else:
        raise Exception('API request failed with status code {}'.format(response.status_code))

# Define the Streamlit app
def app():

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
    if image is not None and st.button('Predict'):

        st.caption("Prediction in progress...")
        # Use st.spinner to display a spinner while predictions are being made
        with st.spinner('Predicting...'):
            # Make API request to get predictions
            predictions = get_predictions(image_bytes)

        # Display the prediction results
        st.write('Prediction:', predictions)

    # If predictions have been made, display a download button for the predictions
    if predictions is not None:
        # Use st.download_button to allow the user to download the predictions as a CSV file
        st.download_button(
            label='Download predictions',
            data=json.dumps(predictions),
            file_name='predictions.json',
            mime='application/json'
        )

# Run the Streamlit app
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", help="endpoint for prediction", default='http://localhost:8000/predict')
    args = parser.parse_args()

    app()
