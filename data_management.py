import gc
import os
import json
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split

class MySQLDatabase:
    def __init__(self, config_path='config.json'):
        """
        初始化資料庫連接
        :param config_path: 配置文件路徑
        """
        self.config_path = config_path
        self.connection = None
        self.cursor = None

    def connect(self):
        """建立資料庫連接並創建 cursor 對象"""
        try:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)
            self.connection = mysql.connector.connect(**config)
            self.cursor = self.connection.cursor()
            print("資料庫連接成功！")
        except mysql.connector.Error as err:
            print(f"資料庫連接失敗: {err}")
            self.connection = None
            self.cursor = None

    def execute_query(self, query, params=None):
        """
        執行 SQL 查詢
        :param query: SQL 查詢語句
        :param params: 查詢參數(可選)
        :return: 查詢結果
        """
        if self.connection and self.cursor:
            try:
                self.cursor.execute(query, params)
                return self.cursor.fetchall()
            except mysql.connector.Error as err:
                print(f"執行時出錯: {err}")
                return None
        else:
            print("未連接到資料庫。")
            return None
        
    def get_features_by_user_id(self, user_id):
        self.connect()  # 建立資料庫連接
        if user_id is not None:
            user_id = int(user_id) # 確保 user_id 是 Python 的 int 類型
            # 查詢 context features
            query_context = "SELECT song_id, source_system_tab, source_screen_name, source_type FROM c_features WHERE user_id=%s"
            context_features = self.execute_query(query_context, (user_id,))

            # 查詢 user features
            query_user = "SELECT number_of_songs, bd, gender FROM u_features WHERE user_id=%s"
            user_features = self.execute_query(query_user, (user_id,))
            
            self.close()  # 關閉資料庫連接
            
            context_feature_df = pd.DataFrame(context_features, columns=['song_id', 'source_system_tab', 'source_screen_name', 'source_type'])
            user_feature_df = pd.DataFrame(user_features, columns=['number_of_songs', 'bd', 'gender'])
            
            # 清理 DataFrame 中的换行符或其他不需要的字符
            user_feature_df['gender'] = user_feature_df['gender'].str.strip()
            
            return context_feature_df, user_feature_df
        
        self.close()  # 關閉資料庫連接（即使 user_id 為 None 也要關閉）
        return pd.DataFrame(), pd.DataFrame()

    def close(self):
        """關閉 cursor 和資料庫連接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("已關閉與資料庫連接。")

class FileManage:
    __paths = {
        # raw data
        'train': r'C:\Users\user\Documents\project\train.csv',
        'songs': r'C:\Users\user\Documents\project\songs.csv',
        'members': r'C:\Users\user\Documents\project\members.csv',
        # encoding data
        'context_data': r'data/context_data.csv',
        'user_data': r'data/user_data.csv',
        'song_data': r'data/song_data.csv',
        'rec_song': r'data/rec_song.csv',
        'train_file': r'D:\project\train_file.csv',
        'test_file': r'D:\project\test_file_filtered.csv',
        # for save_processed_data
        'directory': r'D:\test'
    }
    
    #函式：讀取資料(庫)
    @staticmethod
    def read_raw_files():
        context_feature_df = pd.read_csv(FileManage.__paths['train'], encoding='utf-8')
        song_feature_df = pd.read_csv(FileManage.__paths['songs'], encoding='utf-8')
        user_feature_df = pd.read_csv(FileManage.__paths['members'], encoding='utf-8')
        
        print("文件讀取成功")

        return context_feature_df, song_feature_df[['song_id', 'genre_ids', 'artist_name']], user_feature_df[['msno', 'bd', 'gender']]
    
    #函式：讀取各式檔案
    @staticmethod
    def read_files(user_id=None, file='rec_song'):
        try:
            # 根據 msno 的值返回數據
            if user_id is not None:
                context_feature_df = pd.read_csv(FileManage.__paths['context_data'])
                user_feature_df = pd.read_csv(FileManage.__paths['user_data'])
                
                print("文件讀取成功")
                
                # 設置索引以加快查詢速度
                context_feature_df.set_index('user_id', inplace=True)
                user_feature_df.set_index('user_id', inplace=True)
                
                if user_id in context_feature_df.index:
                    context_feature_df_filtered = context_feature_df.loc[[user_id]]
                else:
                    context_feature_df_filtered = pd.DataFrame(columns=context_feature_df.columns)
                
                if user_id in user_feature_df.index:
                    user_feature_df_filtered = user_feature_df.loc[[user_id]]
                else:
                    user_feature_df_filtered = pd.DataFrame(columns=user_feature_df.columns)
                        
                # 重置索引以便後續操作
                context_feature_df_filtered.reset_index(inplace=True)
                user_feature_df_filtered.reset_index(inplace=True)

                return (context_feature_df_filtered[['song_id', 'source_system_tab', 'source_screen_name', 'source_type']],
                        user_feature_df_filtered)
                
            elif file == 'rec_song':
                rec_song_df = pd.read_csv(FileManage.__paths['rec_song'])
                return rec_song_df
            
            elif file == 'user_data':
                user_data_df = pd.read_csv(FileManage.__paths['user_data'])
                return user_data_df
            
            elif file == 'song_data':
                song_data_df = pd.read_csv(FileManage.__paths['song_data'])
                return song_data_df
            
            elif file == 'train_file':
                train_file_df = pd.read_csv(FileManage.__paths['train_file'])
                return train_file_df
            
            elif file == 'test_file':
                test_file_df = pd.read_csv(FileManage.__paths['test_file'])
                return test_file_df
            
            else:
                raise ValueError(f"無效的 file 參數: {file}")
            
        except FileNotFoundError:
            print("文件未找到，請檢查路徑")

    #函式：儲存每個 batch 放入模型前的資料
    @staticmethod
    def save_processed_data(user_id_repeated, batch_song_ids, batch_predictions, batch_context_features, batch_user_song_feature, index):
        user_id_repeated = user_id_repeated.flatten()
        batch_song_ids = batch_song_ids.flatten()
        batch_predictions = batch_predictions.flatten()

        result_df = pd.DataFrame({
            'user_id': user_id_repeated,
            'song_id': batch_song_ids,
            'source_system_tab_0': batch_context_features[:, 0],
            'source_system_tab_1': batch_context_features[:, 1],
            'source_system_tab_2': batch_context_features[:, 2],
            'source_system_tab_3': batch_context_features[:, 3],
            'source_system_tab_4': batch_context_features[:, 4],
            'source_screen_name_0': batch_context_features[:, 5],
            'source_screen_name_1': batch_context_features[:, 6],
            'source_screen_name_2': batch_context_features[:, 7],
            'source_screen_name_3': batch_context_features[:, 8],
            'source_screen_name_4': batch_context_features[:, 9],
            'source_screen_name_5': batch_context_features[:, 10],
            'source_screen_name_6': batch_context_features[:, 11],
            'source_type_0': batch_context_features[:, 12],
            'source_type_1': batch_context_features[:, 13],
            'source_type_2': batch_context_features[:, 14],
            'source_type_3': batch_context_features[:, 15],
            'source_type_4': batch_context_features[:, 16],
            'source_type_5': batch_context_features[:, 17],
            'number_of_songs': batch_user_song_feature[:, 0],
            'bd': batch_user_song_feature[:, 1],
            'gender_0': batch_user_song_feature[:, 2],
            'gender_1': batch_user_song_feature[:, 3],
            'gender_2': batch_user_song_feature[:, 4],
            '1259': batch_user_song_feature[:, 5],
            '139': batch_user_song_feature[:, 6],
            '1609': batch_user_song_feature[:, 7],
            '2022': batch_user_song_feature[:, 8],
            '2122': batch_user_song_feature[:, 9],
            '359': batch_user_song_feature[:, 10],
            '444': batch_user_song_feature[:, 11],
            '458': batch_user_song_feature[:, 12],
            '465': batch_user_song_feature[:, 13],
            '921': batch_user_song_feature[:, 14],
            'others': batch_user_song_feature[:, 15],
            'support': batch_user_song_feature[:, 16],
            'confidence': batch_user_song_feature[:, 17],
            'artist_support': batch_user_song_feature[:, 18],
            'artist_confidence': batch_user_song_feature[:, 19],
            'prediction': batch_predictions
        })

        file_name = f'results_batch_{index}.csv'
        file_path = os.path.join(FileManage.__paths['directory'], file_name)

        result_df.to_csv(file_path, index=False)

        print(f'保存輸入到 processed_data_batch_{index}.csv 文件')

    # 保存結果到CSV文件
    @staticmethod
    def save_to_csv(df, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(f"目錄 {directory} 已創建")
        if os.path.exists(file_path):
            # 排序數據框架，以確保一致的順序
            df_sorted = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
            
            # 讀取已有的文件
            existing_df = pd.read_csv(file_path)
            existing_df_sorted = existing_df.sort_values(by=existing_df.columns.tolist()).reset_index(drop=True)
                
            # 比較文件內容，包括浮點數的精度
            try:
                pd.testing.assert_frame_equal(df_sorted, existing_df_sorted, check_dtype=False, check_exact=False)
                print(f"文件 {file_path} 已存在且內容相同。")
                return
            except AssertionError:
                # 顯示警告並比較差異
                print(f"警告: 文件 {file_path} 已存在，但內容不同。將會覆蓋文件。")
                
                # 顯示具體差異
                differences = df_sorted.compare(existing_df_sorted)
                print("以下是文件內容的差異:")
                print(differences)
                
                # 提示用戶進行確認
                user_input = input("是否要覆蓋文件？輸入 'y' 來確認，其他任何鍵取消操作: ")
                if user_input.lower() != 'y':
                    print("操作已取消。")
                    return
        else:
            # 文件不存在，直接保存
            print(f"文件 {file_path} 不存在，將創建新文件。")

        # 保存新的CSV文件
        df.to_csv(file_path, index=False)
        print(f"結果已保存到: {file_path}")
        
        
class FileComparer:
    @staticmethod
    def compare_song_ids(original_file, context_file, columns_to_check):
        original_df = pd.read_csv(original_file)
        context_df = pd.read_csv(context_file)

        filtered_df = original_df[original_df[columns_to_check].sum(axis=1) != 0]

        filtered_song_ids = filtered_df['song_id'].unique()
        filtered_user_ids = filtered_df['user_id'].unique()

        matching_songs = context_df[context_df['user_id'].isin(filtered_user_ids)]
        context_song_ids = matching_songs['song_id'].unique()

        common_song_ids = set(filtered_song_ids).intersection(set(context_song_ids))
        unique_filtered_song_ids = set(filtered_song_ids) - common_song_ids
        unique_context_song_ids = set(context_song_ids) - common_song_ids

        print(f"兩個集合中的共同 song_id 數量: {len(common_song_ids)}")

        print("\n在原始資料中，但不在 context_data.csv 中的 song_id:")
        for song_id in unique_filtered_song_ids:
            print(song_id)

        print("\n在 context_data.csv 中，但不在原始資料中的 song_id:")
        for song_id in unique_context_song_ids:
            print(song_id)

    @staticmethod
    def compare_file(df, file_path, compare_columns, precision=6):
        # 確保比較的欄位存在於 DataFrame 中
        missing_cols = [col for col in compare_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame 缺少以下欄位: {', '.join(missing_cols)}")

        # 選擇要比較的欄位
        df_to_compare = df[compare_columns]

        # 確保文件存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在。")
        
        # 設定精度
        pd.set_option('display.precision', precision)

        existing_df = pd.read_csv(file_path)
        existing_df_to_compare = existing_df[compare_columns]
        
        # 調整顯示格式以保留精度
        df_to_compare = df_to_compare.round(precision)
        existing_df_to_compare = existing_df_to_compare.round(precision)

        # 排序數據框架，以確保一致的順序
        #df_to_compare_sorted = df_to_compare.sort_values(by=compare_columns).reset_index(drop=True)
        #existing_df_to_compare_sorted = existing_df_to_compare.sort_values(by=compare_columns).reset_index(drop=True)
        
        print(df_to_compare)
        print(existing_df_to_compare)

        # 比較文件內容
        try:
            pd.testing.assert_frame_equal(df_to_compare, existing_df_to_compare, check_dtype=False, check_exact=False)
            print(f"文件 {file_path} 的內容與新數據相同。")
        except AssertionError:
            # 顯示警告並比較差異
            print(f"警告: 文件 {file_path} 的內容與新數據不同。")

            # 顯示具體差異
            differences = df_to_compare.compare(existing_df_to_compare)
            print("以下是文件內容的差異:")
            print(differences)
        
        
class DataHandler:
    #函式：合併資料
    @staticmethod
    def merge_raw_data(context_feature_df, song_feature_df, user_feature_df):
        merged_df = pd.merge(context_feature_df, song_feature_df, on='song_id', how='left')
        merged_df = pd.merge(merged_df, user_feature_df, on='msno', how='left')
        
        print("合併資料 完成")
        
        return merged_df
    
    @staticmethod
    def drop_rename_col(merged_df, exclude_cols=None, rename_cols=None):
        # 設定預設值
        if exclude_cols is None:
            exclude_cols = ['msno', 'song_id']
        if rename_cols is None:
            rename_cols = {'msno_encoded': 'user_id', 'song_id_encoded': 'song_id'}

        # 排除指定的列
        if not set(exclude_cols).issubset(merged_df.columns):
            missing_cols = list(set(exclude_cols) - set(merged_df.columns))
            raise ValueError(f"以下列不存在於 DataFrame: {missing_cols}")
        
        merged_df.drop(columns=exclude_cols, inplace=True)
        print(f"已排除列: {exclude_cols}")

        # 重命名指定的列
        if not set(rename_cols.keys()).issubset(merged_df.columns):
            missing_rename_cols = list(set(rename_cols.keys()) - set(merged_df.columns))
            raise ValueError(f"無法重命名，因為以下列不存在於 DataFrame: {missing_rename_cols}")
        
        merged_df.rename(columns=rename_cols, inplace=True)
        print(f"已重命名列: {rename_cols}")
        
        return merged_df
    
    # 函数：重新排列列順序
    @staticmethod
    def reorder_columns(merged_df, genre_classes=None):
        if genre_classes is not None:
            genre_columns = list(genre_classes)  # 包含 genre_ids 編碼後
            
            new_order = [
                'user_id', 'song_id', 'target', 
                'source_system_tab', 'source_screen_name', 'source_type', 
                'number_of_songs', 'bd', 'gender'
            ] + genre_columns + [
                'support', 'confidence', 'artist_support', 'artist_confidence'
            ]
            
        else:
            new_order = ['user_id', 'song_id', 
                         'source_system_tab', 'source_screen_name', 'source_type', 
                         'number_of_songs', 'bd', 'gender'
            ]
        # 檢查哪些列不在 DataFrame 中
        missing_columns = [col for col in new_order if col not in merged_df.columns]
        if missing_columns:
            print(f"以下列不存在於 DataFrame 中，將跳過: {missing_columns}")

        # 根據存在的列重新排列
        new_order = [col for col in new_order if col in merged_df.columns]
        merged_df = merged_df.reindex(columns=new_order)
        
        print("已重新排列列順序")
        
        return merged_df
    
    @staticmethod
    # 删除不再需要的 DataFrame 進行垃圾回收。
    def release_memory(*dfs):
        for df in dfs:
            del df
        gc.collect()


class Recommender:
    @staticmethod
    def rec_song(merged_df, path, num_songs, genre_classes):
        rec_song_path = path + r'\rec_song.csv'
        
        genre_columns = list(genre_classes)
        # 保留所需的列
        retained_columns = [
            'song_id_encoded'] + genre_columns + [
            'support', 'confidence', 'artist_support', 'artist_confidence'
        ]
        filtered_df = merged_df.loc[:, retained_columns]
        
        # 去重複
        filtered_df = filtered_df.drop_duplicates(subset=['song_id_encoded'])
        
        # 用song_id_encoded排序
        filtered_df.sort_values(by='song_id_encoded', inplace=True)
        
        # 檢查那些song_id_encoded沒出現過
        all_song_ids = set(range(1, num_songs))
        present_song_ids = set(filtered_df['song_id_encoded'])
        missing_song_ids = all_song_ids - present_song_ids
        if missing_song_ids:
            print(f"未出現的 song_id_encoded: {missing_song_ids}")
        else:
            print("所有 song_id_encoded 都已經出現過。")
            
        filtered_df.rename(columns={'song_id_encoded': 'song_id'}, inplace=True)
        
        # 保存結果到CSV文件
        FileManage.save_to_csv(filtered_df, rec_song_path)


class DataSplitter:
    @staticmethod
    def split_data(merged_df, path, train_size=0.8):
        train_path = path + r'\train_file.csv'
        test_path = path + r'\test_file_filtered.csv'
        
        # 分割資料集，以 4:1 比例，將其分為 train 和 test
        train_df, test_df = train_test_split(merged_df, train_size=train_size, random_state=42)

        # 過濾測試集中的使用者和歌曲，確保它們在訓練集中也存在
        train_users = set(train_df['user_id'])
        train_items = set(train_df['song_id'])

        test_df = test_df[test_df['user_id'].isin(train_users) & test_df['song_id'].isin(train_items)]

        # 儲存處理過後的 train 和 test 資料集到新的 CSV 檔案

        FileManage.save_to_csv(train_df, train_path)
        FileManage.save_to_csv(test_df, test_path)

        return train_df