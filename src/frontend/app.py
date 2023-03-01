import argparse
import streamlit as st
import requests
import sqlite3
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

    # Define session state variables for displaying No button when yes is clicked
    st.session_state["show_no_button"] = True

    # Set app title
    st.title('Animal Classifier')

    # Define a placeholder for the uploaded file data
    image = None

    # Define a placeholder for the predictions
    predictions = None

    # Use st.file_uploader to get a file from the user
    st.caption("Choose an image for classification here")
    file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

    # If a file was uploaded
    if file is not None:

        # Convert image to bytes
        image_bytes = file.read()

        # Display the uploaded image
        image = Image.open(file)
        st.image(image, caption='Uploaded Image')

        predict = st.button('Predict Class')

        if st.session_state.get('button') != True:
            st.session_state['button'] = predict

    # If the user has uploaded an image and wants to make predictions
    if image is not None and st.session_state['button'] == True:

        st.caption("Prediction in progress...")
        # Use st.spinner to display a spinner while predictions are being made
        with st.spinner('Predicting...'):
            # Make API request to get predictions
            predictions = get_predictions(image_bytes)

        # Display the prediction results
        label = predictions["response"]
        st.subheader(f'That animal is a {label}')
        st.write(f"Would you like to view similar images of {label}?")

        if st.button('Yes'):
            st.subheader("Here you go! Enjoy. Click reload to restart.")

            # Connect to the SQLite database
            conn = sqlite3.connect("databases/images.db")
            cursor = conn.cursor()

            # Get a list of 6 random images from the database
            query = f"SELECT file_path FROM metadata WHERE label='{label}' ORDER BY RANDOM() LIMIT 6"
            cursor.execute(query)
            rows = cursor.fetchall()
            images = [row[0] for row in rows]

            # Display the images using Streamlit
            col1, col2, col3 = st.columns(3)

            for i, img_path in enumerate(images):
                image = Image.open(img_path)
                if i % 3 == 0:
                    col1.image(image)
                elif i % 3 == 1:
                    col2.image(image)
                else:
                    col3.image(image)

            # Close DB connection
            conn.close()

            # Session state reverts to false to indicate restart after checking reload
            st.session_state["button"] = False
            st.session_state["show_no_button"] = False

            st.checkbox("Reload")
                
        # Session state to control restart after clicking No
        if st.session_state.get("show_no_button", True):
            if st.button("No"):
                # Clear the image and prediction placeholders
                image = None
                predictions = None

                # Reset the button state
                st.session_state['button'] = False

                # Reload the app
                st.experimental_rerun()


# Run the Streamlit app
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", help="endpoint for prediction", default='http://localhost:8000/predict')
    args = parser.parse_args()

    app()