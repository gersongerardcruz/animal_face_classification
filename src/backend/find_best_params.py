import argparse
import tensorflow as tf
import mlflow, mlflow.keras
import sqlite3
import pandas as pd
from keras import layers, optimizers, models
from keras_tuner.tuners import RandomSearch
from keras.preprocessing.image import ImageDataGenerator
from mlflow.tracking import MlflowClient

# Define the model-building function
def build_model(hp):
    model = models.Sequential()

    # Add the convolutional layers
    model.add(layers.Conv2D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(64, 64, 3)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(
        filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=32),
        kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
        activation='relu',
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Add the dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense1_units', min_value=64, max_value=512, step=64),
        activation='relu'
    ))
    model.add(layers.Dropout(
        rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)
    ))
    model.add(layers.Dense(3, activation='softmax'))

    # Compile the model with the given learning rate and optimizer
    model.compile(
        optimizer=optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3),
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

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
        # Define the hyperparameters to tune
        tuner = RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=args.max_trials,
            executions_per_trial=args.executions_per_trial,
            directory='keras_tuner',
            project_name='animal_classification'
        )

        # Run the hyperparameter search
        tuner.search(
            train,
            validation_data=val,
            epochs=args.tuner_epochs
        )

        # Print the best hyperparameters
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f'Best hyperparameters: {best_hyperparameters}')

        mlflow.keras.autolog(registered_model_name=args.model_name)

        # Build and evaluate the best model
        best_model = tuner.hypermodel.build(best_hyperparameters)
        best_model.fit(
            train,
            validation_data=val,
            epochs=args.epochs
        )

        mlflow.keras.log_model(best_model, "best_model")

        model_uri = mlflow.get_artifact_uri("model")

        print(f"The Keras model is located at: {model_uri}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Name of the mlflow experiment")
    parser.add_argument("--model_name", type=str, help="Name of model to register")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--db_path", type=str, help="Path to database to collect train and val data")
    parser.add_argument("--max_trials", type=int, help="Number of trials to tune with keras-tuner", default=5)
    parser.add_argument("--executions_per_trial", type=int, help="Number of executions per trial", default=3)
    parser.add_argument("--tuner_epochs", type=int, help="Number of trials to tune with keras-tuner", default=10)
    args = parser.parse_args()

    train()