import pandas as pd

# Assuming 'songs_scaled.csv' is your file containing the DataFrame 'songs_scaled'
file_path = './songs_scaled.csv'

# Read the DataFrame from the file
songs_scaled = pd.read_csv(file_path)

# Assuming 'other_file.txt' is the file where you want to save the 'name' column data
output_file = './song_names.txt'

# Retrieve the 'name' column data
name_column = songs_scaled['name']

# Save the 'name' column to another file
name_column.to_csv(output_file, header=False, index=False)  # Change to .to_csv(output_file, header=True, index=False) if you want to include headers
