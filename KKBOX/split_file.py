import pandas as pd

# 讀取 CSV 檔案
file_path = r'D:\song_data.csv'
data = pd.read_csv(file_path)

# 確認資料筆數
total_rows = len(data)
print(f"Total rows: {total_rows}")

# 每個檔案包含的資料筆數
rows_per_file = 10000

# 計算檔案數量
num_files = (total_rows // rows_per_file) + int(total_rows % rows_per_file != 0)
print(f"Number of files to be created: {num_files}")

# 分割並儲存成多個檔案
for i in range(num_files):
    start_row = i * rows_per_file
    end_row = (i + 1) * rows_per_file
    subset = data.iloc[start_row:end_row]
    
    # 檔案名稱加入序號
    file_name = f'output_file_{i + 1:03d}.csv'  # 例如：output_file_001.csv, output_file_002.csv, ...
    subset.to_csv(file_name, index=False)
    print(f"Saved {file_name} with rows from {start_row} to {end_row - 1}")
