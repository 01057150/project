import os
import numpy as np
from tensorflow.keras.models import load_model
from recommendation_model import PredictionProcessor
from data_management import FileManage, MySQLDatabase

# 設置隨機數生成器並生成 380 個不重複的隨機數作為使用者 ID
np.random.seed(42)
user_ids = np.random.choice(range(30755), size=380, replace=False)

# 加載模型
model = load_model(r'D:\best_model_v4.h5')
song_df = FileManage.read_files(file='rec_song')
model.summary()

# 設置輸出目錄
output_dir = r'D:\prediction_results_db'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for user_id in user_ids:
    for read_from_database in [True, False]:
        print(f"Running prediction for user {user_id} with read_from_database = {read_from_database}")
        batch_size = 120000
        
        context_df, user_df = PredictionProcessor.preprocess_data(user_id, read_from_database=read_from_database)

        result_df = PredictionProcessor.predict_for_user(user_id, model, song_df, context_df, user_df, batch_size)

        # 根據 read_from_database 來決定保存的文件名
        if read_from_database:
            file_path = os.path.join(output_dir, f'user_{user_id}_recommendations_mysql.csv')
        else:
            file_path = os.path.join(output_dir, f'user_{user_id}_recommendations_file.csv')
            
        result_df.to_csv(file_path, index=False)
        print(f'Results for user {user_id} saved to {file_path}')
