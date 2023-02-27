import sqlite3
import pandas as pd
import csv
from keras.preprocessing.image import ImageDataGenerator

conn = sqlite3.connect('databases/images.db')
c = conn.cursor()

# Retrieve the image paths and labels from the metadata table
query = 'SELECT file_path, type, label FROM metadata'
c.execute(query)
data = c.fetchall()

# Close the database connection
conn.close()

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data, columns=['file_path', 'type', 'label'])

print(df.head())
print(df.tail())

# Create an instance of the ImageDataGenerator class
datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.5, 1.5], 
    horizontal_flip=True, 
    vertical_flip=True
)

# Use the flow_from_dataframe method to generate the image data and labels
dataflow = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='file_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=len(df),
    class_mode='categorical',
    shuffle=False,
    seed=0
)

# Open the CSV file for writing
with open('data/processed/image_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_path', 'type', 'label'])

    # Iterate over the dataflow object and write the image data and labels to the CSV file
    for images, labels in dataflow:
        for i in range(len(images)):
            writer.writerow([dataflow.filenames[i], df['type'][i], labels[i]])
        break