import argparse
import sqlite3
import pandas as pd
import tensorflow as tf
import mlflow, mlflow.keras
from mlflow.tracking import MlflowClient
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train():
    # Set device to use GPU
    devices = tf.config.list_physical_devices()
    print(devices)
    a=tf.random.normal([100,100])
    b=tf.random.normal([100,100])
    c = a*b

    # Spin up Mlflow client
    client = MlflowClient()

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    # Retrieve the image paths and labels from the metadata table
    query = 'SELECT file_path, type, label FROM metadata'
    cursor.execute(query)
    data = cursor.fetchall()

    # Close the database connection
    conn.close()

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data, columns=['file_path', 'type', 'label'])

    print(df.head())
    print(df.tail())

    train = df[df['type'] == 'train']
    val = df[df['type'] == 'val']

    # Create an instance of the ImageDataGenerator class for train and val
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    # Use the flow_from_dataframe method to generate the image data and labels for train and val
    train = datagen.flow_from_dataframe(
        dataframe=train,
        x_col='file_path',
        y_col='label',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        seed=0
    )

    val = datagen.flow_from_dataframe(
        dataframe=val,
        x_col='file_path',
        y_col='label',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        seed=0
    )

    # Start an MLflow experiment for tracking
    mlflow.set_experiment(args.experiment_name)

    # Start an MLflow run for tracking the Keras experiment
    with mlflow.start_run():

        # create a model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        print(model.summary())

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

        mlflow.keras.autolog(registered_model_name=args.model_name)
        model.fit(train, epochs=args.epochs, steps_per_epoch=train.samples//train.batch_size, validation_data = val, 
                validation_steps=val.samples // val.batch_size, verbose=1)

        mlflow.keras.log_model(model, "model")

        model_uri = mlflow.get_artifact_uri("model")

        print(f"The Keras model is located at: {model_uri}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Name of the mlflow experiment")
    parser.add_argument("--model_name", type=str, help="Name of model to register")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--db_path", type=str, help="Path to database to collect train and val data")
    args = parser.parse_args()

    train()