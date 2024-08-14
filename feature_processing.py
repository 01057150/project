import os
import pickle
import numpy as np
import pandas as pd
from joblib import dump, load
from data_management import FileManage
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer, MultiLabelBinarizer, OneHotEncoder

class FeatureAdder:
    #函式：添加 number_of_songs
    @staticmethod
    def user_features(merged_df):
        # 檢查是否存在 msno 或 user_id 欄位
        if 'msno' in merged_df.columns:
            user_column = 'msno'
        elif 'user_id' in merged_df.columns:
            user_column = 'user_id'
        else:
            raise ValueError("數據框中必須包含 msno 或 user_id 欄位")
        
        # 添加 number_of_songs 欄位，計算每個使用者出現的次數
        merged_df['number_of_songs'] = merged_df.groupby(user_column)[user_column].transform('count')
        
        print("用戶歌曲數 新增完成")
        
        return merged_df
    #函式：添加 support、confidence、artist_support、artist_confidence
    @staticmethod
    def song_features(merged_df, context_feature):

        # 計算支持率和信心度
        song_counts = context_feature['song_id'].value_counts(normalize=True)
        song_target_counts = context_feature[context_feature['target'] == 1]['song_id'].value_counts()
        song_total_counts = context_feature['song_id'].value_counts()

        # 支持率
        merged_df = FeatureAdder.__calculate_support(merged_df, song_counts)

        # 信心度
        merged_df = FeatureAdder.__calculate_confidence(merged_df, song_target_counts, song_total_counts)

        print("歌曲支持率與信心度 新增完成")
        
        # 填充 artist_name 中的 NaN 值
        merged_df.fillna({'artist_name': 'Unknown'}, inplace=True)

        # 計算每個 artist_name 的支持率和信心度
        artist_counts = merged_df.groupby('artist_name').size()
        artist_target_counts = merged_df[merged_df['target'] == 1].groupby('artist_name').size() 

        # 支持率
        merged_df = FeatureAdder.__calculate_artist_support(merged_df, artist_counts)

        # 信心度
        merged_df = FeatureAdder.__calculate_artist_confidence(merged_df, artist_target_counts, artist_counts)

        print("歌手支持率與信心度 新增完成")
        
        return merged_df
    
    @staticmethod
    def __calculate_support(merged_df, song_counts):
        merged_df['support'] = merged_df['song_id'].map(song_counts)
        return merged_df

    @staticmethod
    def __calculate_confidence(merged_df, song_target_counts, song_total_counts):
        merged_df['confidence'] = merged_df['song_id'].map(lambda x: song_target_counts.get(x, 0) / song_total_counts.get(x, 1))
        return merged_df

    @staticmethod
    def __calculate_artist_support(merged_df, artist_counts):
        merged_df['artist_support'] = merged_df['artist_name'].map(lambda x: artist_counts.get(x, 0) / len(merged_df))
        return merged_df

    @staticmethod
    def __calculate_artist_confidence(merged_df, artist_target_counts, artist_counts):
        merged_df['artist_confidence'] = merged_df['artist_name'].map(
            lambda x: artist_target_counts.get(x, 0) / artist_counts.get(x, 1) if artist_counts.get(x, 1) != 0 else 0
        )
        return merged_df

class NumericProcessor:
    # 函数：處理使用者數值特徵
    @staticmethod
    def user_features(merged_df, mode='train', scaler_dir='Scalers', scaler_filename='user_scaler.joblib'):
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        
        scaler_path = os.path.join(scaler_dir, scaler_filename)
        # 定義要處理的數值 col
        numeric_user_cols = ['number_of_songs', 'bd']
            
        # 對數變換高度偏斜的數據
        for col in numeric_user_cols:
            merged_df[col] = np.log1p(merged_df[col])
            print(f"{col} 已進行對數變換")

        if mode == 'train':
            # 創建 StandardScaler 並進行標準化
            scaler = StandardScaler()
            merged_df[numeric_user_cols] = scaler.fit_transform(merged_df[numeric_user_cols])
            
            # 保存 StandardScaler 到文件
            dump(scaler, scaler_path)
            print(f"StandardScaler 已成功保存到 {scaler_path}")
            
        elif mode == 'prediction':
            # 檢查是否需要加載已保存的 StandardScaler
            try:
                scaler = load(scaler_path)
                print(f"已加載 StandardScaler")
                
                # 標準化數據
                merged_df[numeric_user_cols] = scaler.transform(merged_df[numeric_user_cols])
            except FileNotFoundError:
                raise FileNotFoundError(f"找不到已保存的 StandardScaler 文件: {scaler_path}")
            
        else:
            raise ValueError("模式應為 'train' 或 'prediction'")
        
        print("使用者數值特徵已進行標準化")
        print("數據特徵處理完成")
        
        return merged_df
    
    @staticmethod
    def song_features(merged_df, mode='train', scaler_dir='Scalers', scaler_filename='song_scaler.joblib'):
        # 確保 Scalers 資料夾存在
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        
        scaler_path = os.path.join(scaler_dir, scaler_filename)
        # 定義要處理的數值 col
        numeric_song_cols = ['support', 'confidence', 'artist_support', 'artist_confidence']
        
        # 對數變換高度偏斜的數據
        for col in ['support', 'artist_support']:
            merged_df[col] = np.log1p(merged_df[col])
            print(f"{col} 已進行對數變換")
        
        if mode == 'train':
            # 創建 StandardScaler 並進行標準化
            scaler = StandardScaler()
            merged_df[numeric_song_cols] = scaler.fit_transform(merged_df[numeric_song_cols])
            
            # 保存 StandardScaler 到文件
            dump(scaler, scaler_path)
            print(f"StandardScaler 已成功保存到 {scaler_path}")
        
        elif mode == 'prediction':
            # 檢查是否需要加載已保存的 StandardScaler
            try:
                scaler = load(scaler_path)
                print(f"已加載 StandardScaler")

                # 標準化數據
                merged_df[numeric_song_cols] = scaler.transform(merged_df[numeric_song_cols])
            except FileNotFoundError:
                raise FileNotFoundError(f"找不到已保存的 StandardScaler 文件: {scaler_path}")
        
        else:
            raise ValueError("模式應為 'train' 或 'prediction'")
        
        print("歌曲數值特徵已進行標準化")
        print("數據特徵處理完成")
        
        return merged_df
    
    @staticmethod
    def version1(merged_df):
        # number_of_songs 除以最大值縮放到 0~1間
        max_number_of_songs = merged_df['number_of_songs'].max()
        merged_df['number_of_songs'] = merged_df['number_of_songs'] / max_number_of_songs
        print("number_of_songs 已縮放到 0~1 間")

        # bd 使用 Normalize 處理
        normalizer = Normalizer()
        merged_df['bd'] = normalizer.fit_transform(merged_df[['bd']])
        print("bd 已進行 Normalize 處理")
        
        return merged_df
    
    @staticmethod
    def version2(merged_df):
        # 定義要處理的數值 col
        numeric_cols = ['number_of_songs', 'support', 'confidence', 'artist_support', 'artist_confidence', 'bd']
        
        merged_df = NumericProcessor.version1(merged_df)
        
        # 使用 StandardScaler 進行標準化
        scaler = StandardScaler()
        merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])
        
        print("所有數值特徵已進行標準化")
        
        return merged_df

class ContextualProcessor:
    # 函数：處理情境特徵(原始論文方法)
    @staticmethod
    def clean_contextual_features(merged_df):
        #定義情境特徵有效值，注意 source_screen_name 開頭為大寫
        valid_source_system_tab = ['my library', 'discover', 'radio', 'listen with', 'others']
        valid_source_screen_name = ['Local playlist more', 'Online playlist more', 'My library', 'Radio', 'Others profile more', 'Discover Genre', 'Others']
        valid_source_type = ['online-playlist', 'local-playlist', 'local-library', 'radio', 'listen-with', 'others']
        
        # 替換無效值為 'others'
        merged_df['source_system_tab'] = merged_df['source_system_tab'].apply(lambda x: x if x in valid_source_system_tab else 'others')
        merged_df['source_screen_name'] = merged_df['source_screen_name'].apply(lambda x: x if x in valid_source_screen_name else 'Others')
        merged_df['source_type'] = merged_df['source_type'].apply(lambda x: x if x in valid_source_type else 'others')

        print("已處理 contextual_features 中的無效值")
        
        return merged_df
    
    @staticmethod
    def clean_contextual_features_simplify(merged_df):
        #定義情境特徵有效值，注意 source_screen_name 開頭為大寫
        valid_source_system_tab = ['my library', 'discover', 'others']
        valid_source_screen_name = ['Playlist more', 'My library', 'Discover Genre', 'Others']
        valid_source_type = ['online-playlist', 'local-playlist', 'local-library', 'others']
        
        # 替換無效值為 'others'
        merged_df['source_system_tab'] = merged_df['source_system_tab'].apply(lambda x: x if x in valid_source_system_tab else 'others')
        
        # 將 'Local playlist more' 和 'Online playlist more' 變成 'Playlist more'
        merged_df['source_screen_name'] = merged_df['source_screen_name'].apply(lambda x: 'Playlist more' if x in ['Local playlist more', 'Online playlist more'] else (x if x in valid_source_screen_name else 'Others'))
        
        # 將 'online-playlist' 和 'local-playlist' 變成 'playlist'
        merged_df['source_type'] = merged_df['source_type'].apply(lambda x: 'playlist' if x in ['online-playlist', 'local-playlist'] else (x if x in valid_source_type else 'others'))

        print("已處理 contextual_features 中的無效值")
        
        return merged_df

class FeatureProcessor:
    # 函数：處理 genre_ids，注意到複數會用'|'隔開
    @staticmethod
    def genre_ids(merged_df):
        # 定義歌曲種類有效值
        valid_genre_ids = ['465', '458', '921', '1609', '444', '1259', '2022', '359', '139', '2122', 'others']
        
        def replace_invalid_genres(genre_str):
            genres = genre_str.split('|')
            replaced_genres = [genre if genre in valid_genre_ids else 'others' for genre in genres]
            return replaced_genres
        
        # 替換無效值為 'others' 並將 genre_ids 列處理為列表
        merged_df['genre_ids'] = merged_df['genre_ids'].astype(str).apply(replace_invalid_genres)
        
        print("已處理 genre_ids_features 中的無效值")
        
        # 使用 MultiLabelBinarizer 進行編碼
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(merged_df['genre_ids'])
        
        # 將編碼結果轉為 DataFrame
        genre_encoded_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
        
        # 合併原始數據框和編碼後的 genre_ids
        merged_df = merged_df.join(genre_encoded_df)
        
        # 移除原始的 genre_ids 列
        merged_df.drop(columns=['genre_ids'], inplace=True)
        
        print("已完成 genre_ids 的 OneHotEncoder")
        
        return merged_df, mlb.classes_
    
    #函式：處理性別與年齡
    @staticmethod
    def gender_bd_features(user_feature_df):
        # 填充 gender 中的空值
        user_feature_df.fillna({'gender': 'none'}, inplace=True)

        # 檢查和替換 bd
        # 計算 14 到 68 之間的平均年齡
        valid_bd = user_feature_df['bd'][(user_feature_df['bd'] >= 14) & (user_feature_df['bd'] <= 68)]
        average_bd = valid_bd.mean()

        # 替換不在 14 到 68 範圍內的 bd
        user_feature_df['bd'] = user_feature_df['bd'].apply(lambda x: average_bd if x < 14 or x > 68 else x)

        print("gender 與 bd 修改完成")
        
        return user_feature_df
    
    @staticmethod
    def one_hot_encoder(train_df, mode='train'):
        user_ids = train_df['user_id'].values
        song_ids = train_df['song_id'].values
            
        genre_columns = ['465', '458', '921', '1609', '444', '1259', '2022', '359', '139', '2122', 'others']
        genre_features_encoded = train_df[genre_columns].values

        contextual_features_encoded = FeatureProcessor.one_hot_encode_contextual_features(
            train_df[['source_system_tab', 'source_screen_name', 'source_type']], mode = mode
        )

        print("合併後的 contextual_features 前幾行:")
        print(pd.DataFrame(contextual_features_encoded).head())
            
        gender_features_encoded = FeatureProcessor.one_hot_encode_gender_features(train_df[['gender']], mode = mode)

        numeric_song_features = train_df[['support', 'confidence', 'artist_support', 'artist_confidence']].values
        numeric_user_features = train_df[['number_of_songs', 'bd']].values
        
        user_song_features = np.concatenate([
            numeric_user_features,
            gender_features_encoded,
            genre_features_encoded,
            numeric_song_features
        ], axis=1)

        print("合併後的 user_song_features 前幾行:")
        print(pd.DataFrame(user_song_features).head())

        targets = train_df['target'].values

        x_train = [user_ids, song_ids, contextual_features_encoded, user_song_features]
        y_train = targets
            
        print('數據準備完成')
        return x_train, y_train
    
    @staticmethod
    def one_hot_encode_contextual_features(df, mode='train', encoder_dir='OneHotEncoder', encoder_filename='contextual_encoder.joblib'):
        if not os.path.exists(encoder_dir):
            os.makedirs(encoder_dir)
    
        encoder_path = os.path.join(encoder_dir, encoder_filename)
        contextual_features = df[['source_system_tab', 'source_screen_name', 'source_type']]
        
        if mode == 'train':
            encoder_contextual = OneHotEncoder(sparse_output=False)
            encoded_features = encoder_contextual.fit_transform(contextual_features)
            
            # 保存編碼器
            dump(encoder_contextual, encoder_path)
            print("上下文特徵編碼器已保存")

        elif mode in ['prediction', 'test']:
            try:
                encoder_contextual = load(encoder_path)
                print("已加載情境特徵編碼器")
                encoded_features = encoder_contextual.transform(contextual_features)
            except FileNotFoundError:
                raise FileNotFoundError("未找到上下文特徵編碼器文件。請確保在訓練階段已經保存了編碼器。")
        else:
            raise ValueError("模式應為 'train' 或 'prediction' 或 'test'")
            
        return encoded_features
    
    @staticmethod
    def one_hot_encode_gender_features(df, mode='train', encoder_dir='OneHotEncoder', encoder_filename='gender_encoder.joblib'):
        if not os.path.exists(encoder_dir):
            os.makedirs(encoder_dir)
    
        encoder_path = os.path.join(encoder_dir, encoder_filename)
        gender_features = df[['gender']]
        
        if mode == 'train':
            encoder_gender = OneHotEncoder(sparse_output=False)
            encoded_features = encoder_gender.fit_transform(gender_features)

            # 保存編碼器
            dump(encoder_gender, encoder_path)
            print("性別特徵編碼器已保存")

        elif mode in ['prediction', 'test']:
            try:
                encoder_gender = load(encoder_path)
                print("已加載性別特徵編碼器")
                encoded_features = encoder_gender.transform(gender_features)
            except FileNotFoundError:
                raise FileNotFoundError("未找到性別特徵編碼器文件。請確保在訓練階段已經保存了編碼器。")
        else:
            raise ValueError("模式應為 'train' 或 'prediction' 或 'test'")

        return encoded_features

class Encoder:
    # 函数：編碼 msno、song_id
    @staticmethod
    def encode(merged_df, path):
        # 創建 LabelEncoder 實例
        msno_encoder = LabelEncoder()
        song_id_encoder = LabelEncoder()
        merged_df['msno_encoded'] = msno_encoder.fit_transform(merged_df['msno'])
        merged_df['song_id_encoded'] = song_id_encoder.fit_transform(merged_df['song_id'])
        
        # 保存對照表
        msno_mapping = pd.DataFrame({
            'msno': msno_encoder.classes_,
            'msno_encoded': range(len(msno_encoder.classes_))
        })
        
        song_id_mapping = pd.DataFrame({
            'song_id': song_id_encoder.classes_,
            'song_id_encoded': range(len(song_id_encoder.classes_))
        })
        
        # 保存 LabelEncoder
        Encoder.save_label_encoders(msno_encoder, song_id_encoder, path)
        
        # 計算維度
        #num_users = msno_mapping.max()
        #num_songs = song_id_mapping.max()
        num_users = len(msno_mapping)
        num_songs = len(song_id_mapping)
        print(f"使用者人數：{num_users}")
        print(f"歌曲數量：{num_songs}")
        
        # 保存到 CSV 文件
        FileManage.save_to_csv(msno_mapping, path + r'\msno_mapping.csv')
        FileManage.save_to_csv(song_id_mapping, path + r'\song_id_mapping.csv')

        return merged_df, num_users, num_songs
    
    # 保存 LabelEncoder 實例，包含了所有 user_id song_id 的映射關係
    @staticmethod
    def save_label_encoders(msno_encoder, song_id_encoder, path):
        with open(path + r'\msno_encoder.pkl', 'wb') as f:
            pickle.dump(msno_encoder, f)
        with open(path + r'\song_id_encoder.pkl', 'wb') as f:
            pickle.dump(song_id_encoder, f)

    # 讀取 LabelEncoder 實例
    @staticmethod
    def load_label_encoders(path):
        with open(path + r'\msno_encoder.pkl', 'rb') as f:
            msno_encoder = pickle.load(f)
        with open(path + r'\song_id_encoder.pkl', 'rb') as f:
            song_id_encoder = pickle.load(f)
        return msno_encoder, song_id_encoder

    # 讀取 LabelEncoder 實例
    @staticmethod
    def update_label_encoders(encoder, new_values):
        new_classes = pd.Series(new_values).unique()
        current_classes = encoder.classes_
        updated_classes = pd.Series(current_classes).append(pd.Series(new_classes)).unique()
        encoder.classes_ = updated_classes
        return encoder