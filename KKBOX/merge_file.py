import pandas as pd
import os

directory = r'D:'
file_prefix = 'song_data_extend_'
file_suffix = '.csv'
file_range = range(1, 37)  
# 合并所有文件
dfs = []
for i in file_range:
    file_number = f'{i:03}'
    file_path = os.path.join(directory, f'{file_prefix}{file_number}{file_suffix}')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        dfs.append(df)
    else:
        print(f"文件 {file_path} 不存在")

merged_df = pd.concat(dfs, ignore_index=True)

empty_track_id_count = merged_df['track_id'].isna().sum()  
total_count = len(merged_df)
empty_track_id_ratio = empty_track_id_count / total_count if total_count > 0 else 0

print(f"track_id 為空的數量: {empty_track_id_count}")
print(f"track_id 為空的比例: {empty_track_id_ratio:.4f}")

output_file = r'D:\merged_song_data.csv'
merged_df.to_csv(output_file, index=False)
print(f"合併後的文件以保存為 {output_file}")
