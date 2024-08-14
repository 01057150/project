import pandas as pd
import os
from data_management import FileComparer

# 指定包含CSV檔案的資料夾路徑
folder_path = r'D:\test'
compare_file = r'D:\project\rec_song.csv'
original_file = r'D:\test\merged_results.csv'
context_file = r'D:\context_data.csv'

# 找到所有符合條件的CSV檔案(results_batch_{index})
files = [f for f in os.listdir(folder_path) if f.startswith('results_batch_') and f.endswith('.csv')]
files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))  # 根據index排序

# 讀取並合併所有CSV檔案
dfs = [pd.read_csv(os.path.join(folder_path, file)) for file in files]
merged_df = pd.concat(dfs, ignore_index=True)

# 儲存合併後的結果
output_file = os.path.join(folder_path, 'merged_results.csv')
merged_df.to_csv(output_file, index=False)

print(f'所有檔案已成功合併並儲存在: {output_file}')

context_df = pd.read_csv(r'D:\context_data.csv')

# 定義要比較的欄位
compare_columns = ['1259', '139', '1609', '2022', '2122', '359', '444', '458', '465', '921', 'others', 
                   'support', 'confidence', 'artist_support', 'artist_confidence']

# 定義需要檢查的欄位
columns_to_check = [
    'source_system_tab_0', 'source_system_tab_1', 'source_system_tab_2', 'source_system_tab_3', 'source_system_tab_4',
    'source_screen_name_0', 'source_screen_name_1', 'source_screen_name_2', 'source_screen_name_3', 
    'source_screen_name_4', 'source_screen_name_5', 'source_screen_name_6',
    'source_type_0', 'source_type_1', 'source_type_2', 'source_type_3', 'source_type_4', 'source_type_5'
]

FileComparer.compare_file(merged_df, compare_file, compare_columns)
FileComparer.compare_song_ids(original_file, context_file, columns_to_check)

