from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from feature_processing import NumericProcessor, ContextualProcessor, FeatureProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_management import FileManage
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import threading
import time
import os

class MLPModelBuilder:
    def __init__(self, num_users, num_songs, user_feature_dim, song_feature_dim, contextual_feature_dim, user_embedding_dim, song_embedding_dim):
        self.num_users = num_users
        self.num_songs = num_songs
        self.user_feature_dim = user_feature_dim
        self.song_feature_dim = song_feature_dim
        self.contextual_feature_dim = contextual_feature_dim
        self.user_embedding_dim = user_embedding_dim
        self.song_embedding_dim = song_embedding_dim
    
    def create_model(self):
        user_id_input = Input(shape=(1,), name='user_id')
        song_id_input = Input(shape=(1,), name='song_id')

        user_embedding = Embedding(input_dim=self.num_users + 1, output_dim=self.user_embedding_dim, name='user_embedding', embeddings_regularizer=l2(1e-6))(user_id_input)
        song_embedding = Embedding(input_dim=self.num_songs + 1, output_dim=self.song_embedding_dim, name='song_embedding', embeddings_regularizer=l2(1e-6))(song_id_input)

        user_embedding = Flatten()(user_embedding)
        song_embedding = Flatten()(song_embedding)

        user_embedding = Dense(64, activation='relu')(user_embedding)
        user_embedding = BatchNormalization()(user_embedding)
        song_embedding = Dense(128, activation='relu')(song_embedding)
        song_embedding = BatchNormalization()(song_embedding)

        contextual_features_input = Input(shape=(self.contextual_feature_dim,), name='contextual_features')
        user_song_features_input = Input(shape=(self.user_feature_dim + self.song_feature_dim,), name='user_song_features')

        contextual_feature_embedding = Dense(64, activation='relu')(contextual_features_input)
        contextual_feature_embedding = BatchNormalization()(contextual_feature_embedding)
        user_song_feature_embedding = Dense(64, activation='relu')(user_song_features_input)
        user_song_feature_embedding = BatchNormalization()(user_song_feature_embedding)

        concatenated = Concatenate()([user_embedding, song_embedding, contextual_feature_embedding, user_song_feature_embedding])

        dense1 = Dense(512, activation='relu')(concatenated)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.5)(dense1)
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.5)(dense2)
        output = Dense(1, activation='sigmoid')(dense2)

        model = Model(inputs=[user_id_input, song_id_input, contextual_features_input, user_song_features_input], outputs=output)
        return model

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, x_train, y_train):
        model_name = 'model'
        model_path = os.path.join(self.config['checkpoint_path'], model_name + '.h5')
        early_stop = EarlyStopping(monitor='val_loss', patience=self.config['patience'], mode='min')
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        update_pruning = tfmot.sparsity.keras.UpdatePruningStep()
        
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            callbacks=[early_stop, checkpoint, update_pruning],
            verbose=1
        )
        return history

class PredictionProcessor:
    @staticmethod
    def predict_for_user(user_id, model, song_df, context_df, user_df, batch_size=1000):
        """
        為給定的用戶進行預測。
        
        參數:
        user_id (str): 用戶ID。
        model (Model): 預測模型。
        batch_size (int): 批次大小。
        
        返回:
        DataFrame: 按預測結果排序的結果數據框。
        """
        predictions = []
        all_song_ids = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, 359966, batch_size):
                futures.append(executor.submit(PredictionProcessor.__preprocess_and_predict_batch, 
                                               user_id, song_df, context_df, user_df, model, i, batch_size))
            
            for future in as_completed(futures):
                batch_song_ids, batch_predictions = future.result()
                all_song_ids.extend(batch_song_ids)
                predictions.extend(batch_predictions)
        
        # 將預測結果組合成 DataFrame，並根據預測結果排序
        result_df = pd.DataFrame({
            'song_id': np.concatenate(all_song_ids),
            'prediction': np.concatenate(predictions)
        })
        result_df = result_df.sort_values(by='prediction', ascending=False)
        
        return result_df
    
    @staticmethod
    def preprocess_data(user_id):
        context_df, user_df = FileManage.read_files(user_id=user_id)
        context_df = ContextualProcessor.clean_contextual_features(context_df)
        context_df_encode = FeatureProcessor.one_hot_encode_contextual_features(context_df, mode='prediction')
        user_df = NumericProcessor.user_features(user_df, mode='prediction')
        gender_encode = FeatureProcessor.one_hot_encode_gender_features(user_df, mode='prediction')
        context_df = pd.concat([context_df[['song_id']], pd.DataFrame(context_df_encode)], axis=1)
        user_df = pd.concat([user_df[['number_of_songs', 'bd']], pd.DataFrame(gender_encode)], axis=1)
        return context_df, user_df
    
    @staticmethod
    def __preprocess_and_predict_batch(user_id, song_df, context_df, user_df, model, start_index, batch_size):
        """
        預處理並預測批次數據。
        
        參數:
        user_id (str): 用戶ID。
        song_df (DataFrame): 批次歌曲數據。
        context_df (DataFrame): 批次上下文數據。
        user_features (DataFrame): 用戶特徵。
        model (Model): 預測模型。
        start_index (int): 批次索引。
        
        返回:
        tuple: 批次歌曲ID和預測結果。
        """
        batch_song_ids = song_df['song_id'].values[start_index:start_index + batch_size]
        batch_song_ids_df = pd.DataFrame(batch_song_ids, columns=['song_id'])
        
        batch_context_df = pd.merge(batch_song_ids_df, context_df, on='song_id', how='left')
        batch_context_df = batch_context_df.fillna(0)
        
        batch_song_df = song_df[song_df['song_id'].isin(batch_song_ids)].drop(columns=['song_id'])
        
        batch_song_ids = batch_context_df[['song_id']].values
        batch_context_features = batch_context_df.drop(columns=['song_id']).values
        
        user_features_repeated = np.tile(user_df.values, (batch_context_features.shape[0], 1))
        user_id_repeated = np.tile(user_id, (batch_context_features.shape[0], 1))
        
        batch_user_song_feature = np.concatenate([user_features_repeated, batch_song_df.values], axis=1)
        x_prediction = [user_id_repeated, batch_song_ids, batch_context_features, batch_user_song_feature]
        
        with tf.device('/gpu:0'):
            batch_predictions = model.predict(x_prediction, batch_size = batch_size)
        #將結果儲存起來
        #FileManage.save_processed_data(user_id_repeated, batch_song_ids, batch_predictions, batch_context_features, batch_user_song_feature, start_index)
        return batch_song_ids, batch_predictions
    
    @staticmethod
    def predict_for_user_time(user_id, model, batch_size=1000):
        """
        在 predict_for_user 加入計時，測試使用。
        
        參數:
        user_id (str): 用戶ID。
        model (Model): 預測模型。
        batch_size (int): 批次大小。
        
        返回:
        DataFrame: 按預測結果排序的結果數據框。
        """
        _time_lock = threading.Lock()
        total_preparation_duration = 0
        total_prediction_duration = 0
        num_batches = 0
        
        start_time = time.time()
        
        context_df, user_df = PredictionProcessor.__preprocess_data_time(user_id)
        song_df = FileManage.read_files(file='rec_song')
        
        data_prep_time = time.time()
        print(f"{'===== Data preparation took':<38} {data_prep_time - start_time:>6.4f} seconds =====")

        predictions = []
        all_song_ids = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, 359966, batch_size):
                futures.append(executor.submit(PredictionProcessor.__preprocess_and_predict_batch_time, 
                                               user_id, song_df, context_df, user_df, model, i, batch_size))
            
            for future in as_completed(futures):
                batch_song_ids, batch_predictions, preparation_duration, prediction_duration= future.result()
                all_song_ids.extend(batch_song_ids)
                predictions.extend(batch_predictions)
                
                with _time_lock:
                    total_preparation_duration += preparation_duration
                    total_prediction_duration += prediction_duration
                num_batches += 1
                

        average_prep_time = total_preparation_duration / num_batches
        average_predict_time = total_prediction_duration / num_batches
        print(f"{'      num_batches:':<38} {num_batches:>6} batches      ")
        print(f"{'      Average batch preparation time:':<38} {average_prep_time:>6.4f} seconds      ")
        print(f"{'      Average batch prediction time:':<38} {average_predict_time:>6.4f} seconds      ")
                
        result_prep_time = time.time()
        print(f"{'===== Result preparation took':<38} {result_prep_time - data_prep_time:>6.4f} seconds =====")
        
        
        result_df = pd.DataFrame({
            'song_id': np.concatenate(all_song_ids),
            'prediction': np.concatenate(predictions)
        })
        result_df = result_df.sort_values(by='prediction', ascending=False)
        
        end_time = time.time()
        print(f"{'===== Total predict_for_user took':<38} {end_time - start_time:>6.4f} seconds =====")
        
        return result_df, (data_prep_time - start_time), average_prep_time, average_predict_time, (result_prep_time - data_prep_time), (end_time - start_time)
    
    @staticmethod
    def __preprocess_data_time(user_id):
        start_time = time.time()
        
        context_df, user_df = FileManage.read_files(user_id=user_id)
        read_time = time.time()
        print(f"{'      Reading files took':<38} {read_time - start_time:>6.4f} seconds")
        
        context_df = ContextualProcessor.clean_contextual_features(context_df)
        context_df_encode = FeatureProcessor.one_hot_encode_contextual_features(context_df, mode='prediction')
        
        user_df = NumericProcessor.user_features(user_df, mode='prediction')
        gender_encode = FeatureProcessor.one_hot_encode_gender_features(user_df, mode='prediction')

        context_df = pd.concat([context_df[['song_id']], pd.DataFrame(context_df_encode)], axis=1)
        user_df = pd.concat([user_df[['number_of_songs', 'bd']], pd.DataFrame(gender_encode)], axis=1)
        
        process_file_time = time.time()
        print(f"{'      Processing files took':<38} {process_file_time - read_time:>6.4f} seconds")
        return context_df, user_df
    
    @staticmethod
    def __preprocess_and_predict_batch_time(user_id, song_df, context_df, user_df, model, start_index, batch_size):
        
        batch_start_time = time.time()
        
        batch_song_ids = song_df['song_id'].values[start_index:start_index + batch_size]
        batch_song_ids_df = pd.DataFrame(batch_song_ids, columns=['song_id'])
        
        batch_context_df = pd.merge(batch_song_ids_df, context_df, on='song_id', how='left')
        batch_context_df = batch_context_df.fillna(0)
        
        batch_song_df = song_df[song_df['song_id'].isin(batch_song_ids)].drop(columns=['song_id'])
        
        batch_song_ids = batch_context_df[['song_id']].values
        batch_context_features = batch_context_df.drop(columns=['song_id']).values
        
        user_features_repeated = np.tile(user_df.values, (batch_context_features.shape[0], 1))
        user_id_repeated = np.tile(user_id, (batch_context_features.shape[0], 1))
        
        batch_user_song_feature = np.concatenate([user_features_repeated, batch_song_df.values], axis=1)
        x_prediction = [user_id_repeated, batch_song_ids, batch_context_features, batch_user_song_feature]
        
        batch_prepare_time = time.time()
        preparation_duration = batch_prepare_time - batch_start_time
        print(f"      {start_index:06} batch preparation took {preparation_duration:>9.4f} seconds")
        
        with tf.device('/gpu:0'):
            batch_predictions = model.predict(x_prediction, batch_size = batch_size)
            
        batch_predict_time = time.time()
        prediction_duration = batch_predict_time - batch_prepare_time
        print(f"      {start_index:06} batch prediction took {prediction_duration:>10.4f} seconds")
        
        # Return the batch song IDs, predictions, preparation time, and prediction time
        return batch_song_ids, batch_predictions, preparation_duration, prediction_duration

class ResultsHandler:
    def __init__(self, model, history, path='.', model_name='model'):
        self.model = model
        self.history = history
        self.path = path
        self.model_name = model_name

    def save_results(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        model_path = os.path.join(self.path, self.model_name + '.h5')
        self.model.save(model_path)
        print(f'模型已保存為 {model_path}')

        history_path = os.path.join(self.path, 'training_history.csv')
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(history_path, index=False)
        print(f"訓練數據已保存為 {history_path}")

    def plot_results(self, plot_path=None, plot_show=False):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['auc'], label='Training AUC')
        plt.plot(self.history.history['val_auc'], label='Validation AUC')
        plt.title('AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()

        plot_path = os.path.join(self.path, 'training_plot.png')
        plt.savefig(plot_path)
        print(f"圖表已保存為 {plot_path}")
        
        if plot_show:
            plt.show()