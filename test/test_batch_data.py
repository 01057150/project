import pandas as pd
import os
from my_package import FileComparer

# Specify the folder path containing the CSV files
folder_path = r'D:\test'
compare_file = r'D:\project\rec_song.csv'
original_file = r'D:\test\merged_results.csv'
context_file = r'D:\context_data.csv'

# Find all CSV files matching the pattern 'results_batch_{index}.csv'
files = [f for f in os.listdir(folder_path) if f.startswith('results_batch_') and f.endswith('.csv')]
files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))  # Sort by index

# Read and merge all CSV files
dfs = [pd.read_csv(os.path.join(folder_path, file)) for file in files]
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged results
output_file = os.path.join(folder_path, 'merged_results.csv')
merged_df.to_csv(output_file, index=False)

print(f'All files have been successfully merged and saved to: {output_file}')

# Read context data file
context_df = pd.read_csv(context_file)

# Define the columns to compare
compare_columns = ['1259', '139', '1609', '2022', '2122', '359', '444', '458', '465', '921', 'others', 
                   'support', 'confidence', 'artist_support', 'artist_confidence']

# Define the columns to check in the context data
columns_to_check = [
    'source_system_tab_0', 'source_system_tab_1', 'source_system_tab_2', 'source_system_tab_3', 'source_system_tab_4',
    'source_screen_name_0', 'source_screen_name_1', 'source_screen_name_2', 'source_screen_name_3', 
    'source_screen_name_4', 'source_screen_name_5', 'source_screen_name_6',
    'source_type_0', 'source_type_1', 'source_type_2', 'source_type_3', 'source_type_4', 'source_type_5'
]

# Compare the merged results with the comparison file
FileComparer.compare_file(merged_df, compare_file, compare_columns)

# Compare song IDs in the original file with the context data file
FileComparer.compare_song_ids(original_file, context_file, columns_to_check)

