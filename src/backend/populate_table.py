import csv
import sqlite3

# Define directories containing CSV files to be read
directories = ["data/raw/train.csv", "data/raw/val.csv"]

# Initialize empty list to store rows of CSV data
rows = []

# Loop through each directory and read the CSV file contents
id_counter = 0
for i, directory in enumerate(directories):
    with open(directory, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            rows.append((id_counter+1,) + tuple(row))  # add id column based on directory index
            id_counter += 1

# Connect to database
conn = sqlite3.connect('databases/images.db')
c = conn.cursor()

# Define table schema and create table if it doesn't exist
create_table_query = '''
CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY,
    file_name TEXT,
    file_path TEXT,
    bytes_size INTEGER,
    resolution TEXT,
    aspect_ratio REAL,
    label TEXT
);
'''
c.execute(create_table_query)

# Insert read rows into database
insert_query = 'INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)'
c.executemany(insert_query, rows)

# Commit changes and close database connection
conn.commit()
conn.close()
