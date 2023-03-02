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

# Define table schema
table_schema = '''
    id INTEGER PRIMARY KEY,
    file_name TEXT,
    file_path TEXT,
    bytes_size INTEGER,
    resolution TEXT,
    aspect_ratio REAL,
    type TEXT,
    label TEXT
'''

# Check if table exists
check_table_query = '''
    SELECT count(name) FROM sqlite_master WHERE type='table' AND name='metadata'
'''
conn = sqlite3.connect('databases/images.db')
c = conn.cursor()
c.execute(check_table_query)
table_exists = c.fetchone()

# Create or update table
if table_exists[0] == 1:
    # Clear existing table contents
    clear_table_query = '''
        DELETE FROM metadata
    '''
    c.execute(clear_table_query)
    conn.commit()

else:
    # Create new table
    create_table_query = f'''
        CREATE TABLE metadata ({table_schema})
    '''
    c.execute(create_table_query)
    conn.commit()

# Insert read rows into database
insert_query = 'INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
c.executemany(insert_query, rows)

conn.commit()
conn.close()