import pandas as pd
import os

# Define file paths and range of user IDs
base_path = r'D:\prediction_results'
user_id_range = range(0, 30755)  # User IDs from 0 to 30754
log_file_path = r'D:\prediction_results\comparison_log.txt'  # Log file path

# Set float display format and precision for comparison
pd.set_option('display.float_format', lambda x: f'{x:.8f}')

# Initialize counters for correct and different files
correct_count = 0
different_count = 0
    
# Open the log file for writing
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    for user_id in user_id_range:
        # Construct file paths
        file1_path = os.path.join(base_path, f'user_{user_id}_recommendations_nBS.csv')
        file2_path = os.path.join(base_path, f'user_{user_id}_recommendations_mysql.csv')
        
        # Check if files exist
        if not os.path.isfile(file1_path) or not os.path.isfile(file2_path):
            log_file.write(f"File does not exist: {file1_path} or {file2_path}\n")
            continue
        
        log_file.write(f"Comparing: {file1_path} and {file2_path}\n")

        try:
            # Read CSV files
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)
            
            # Sort DataFrames by 'song_id' and reset index
            df1_sorted = df1.sort_values(by=['song_id']).reset_index(drop=True)
            df2_sorted = df2.sort_values(by=['song_id']).reset_index(drop=True)
            
            # Round the values to 8 decimal places for comparison
            df1_sorted = df1_sorted.round(8)
            df2_sorted = df2_sorted.round(8)
            
            # Compare DataFrames
            pd.testing.assert_frame_equal(df1_sorted, df2_sorted, check_dtype=False, check_exact=False)
            log_file.write(f"Contents are the same: {file1_path} and {file2_path}\n")
            correct_count += 1
        
        except AssertionError:
            log_file.write(f"File contents differ: {file1_path} and {file2_path}\n")
            differences = df1_sorted.compare(df2_sorted)
            log_file.write(differences.to_string() + "\n")
            different_count += 1

# Print summary at the end
summary_message = f"Compared {correct_count + different_count} file pairs, with {correct_count} correct and {different_count} different."
print(summary_message)

# Append summary to the log file
with open(log_file_path, 'a', encoding='utf-8') as log_file:
    log_file.write(summary_message + "\n")
