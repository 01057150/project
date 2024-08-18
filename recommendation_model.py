from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from feature_processing import NumericProcessor, ContextualProcessor, FeatureProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_management import FileManage, MySQLDatabase
from tensorflow.keras.utils import plot_model
from datetime import datetime
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
        model_path = os.path.join(self.config['checkpoint_path'], self.config['model_name'] + '_checkpoint' + '.h5')
        early_stop = EarlyStopping(monitor='val_loss', patience=self.config['patience'], mode='min')
        checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        return history

class PredictionProcessor:
    @staticmethod
    def predict_for_user(user_id, model, song_df, context_df, user_df, batch_size=1000, set_batch_size=None, save_batch_file=False):
        """
        Make predictions for a given user.

        Parameters:
        user_id (str): User ID.
        model (tf.keras.Model): Prediction model.
        song_df (pd.DataFrame): DataFrame containing song data.
        context_df (pd.DataFrame): DataFrame containing contextual features.
        user_df (pd.DataFrame): DataFrame containing user features.
        batch_size (int): Batch size for predictions.
        set_batch_size (int, optional): Determines which prediction method to use based on its value:
            - If `True`, use `model.predict` with the specified batch size.
            - If `False`, use `model.predict`.
            - Otherwise, use `model.predict_on_batch`.(defalut)
        save_batch_file (bool, optional): If True, save each batch to a file. Default is False.

        Returns:
        pd.DataFrame: DataFrame of results sorted by prediction score.
        """
        predictions = []
        all_song_ids = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, 359966, batch_size):
                futures.append(executor.submit(PredictionProcessor.__preprocess_and_predict_batch, 
                                               user_id, song_df, context_df, user_df, model, i, batch_size, set_batch_size, save_batch_file))
            
            for future in as_completed(futures):
                batch_song_ids, batch_predictions = future.result()
                all_song_ids.extend(batch_song_ids)
                predictions.extend(batch_predictions)
        
        # Combine predictions into a DataFrame and sort by prediction score
        result_df = pd.DataFrame({
            'song_id': np.concatenate(all_song_ids),
            'prediction': np.concatenate(predictions)
        })
        result_df = result_df.sort_values(by='prediction', ascending=False)
        
        return result_df
    
    @staticmethod
    def preprocess_data(user_id, read_from_database=False):
        """
        Preprocess data for the given user.

        Parameters:
        user_id (str): User ID.
        read_from_database (bool): Flag indicating whether to read data from the database.

        Returns:
        tuple: Processed context_df and user_df DataFrames.
        """
        if read_from_database:
            context_df, user_df = MySQLDatabase().get_features_by_user_id(user_id)
        else:
            context_df, user_df = FileManage.read_files(user_id=user_id)
        context_df = ContextualProcessor.clean_contextual_features(context_df)
        context_df_encode = FeatureProcessor.one_hot_encode_contextual_features(context_df, mode='prediction')
        user_df = NumericProcessor.user_features(user_df, mode='prediction')
        gender_encode = FeatureProcessor.one_hot_encode_gender_features(user_df, mode='prediction')
        context_df = pd.concat([context_df[['song_id']], pd.DataFrame(context_df_encode)], axis=1)
        user_df = pd.concat([user_df[['number_of_songs', 'bd']], pd.DataFrame(gender_encode)], axis=1)
        return context_df, user_df
    
    @staticmethod
    def __preprocess_and_predict_batch(user_id, song_df, context_df, user_df, model, start_index, batch_size, set_batch_size, save_batch_file=False):
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
        
        # Choose the prediction method
        predict_method = (model.predict_on_batch if set_batch_size is None else
                          model.predict if set_batch_size is False else
                          lambda x: model.predict(x, batch_size=batch_size))
        
        with tf.device('/gpu:0'):
            batch_predictions = predict_method(x_prediction)

        if save_batch_file:
            FileManage.save_processed_data(user_id_repeated, batch_song_ids, batch_predictions, batch_context_features, batch_user_song_feature, start_index)
        
        return batch_song_ids, batch_predictions
    
    @staticmethod
    def predict_for_user_time(user_id, model, song_df, context_df, user_df, batch_size=1000, set_batch_size=None, save_batch_file=False):
        """
        Make predictions for a given user.

        Parameters:
        user_id (str): User ID.
        model (tf.keras.Model): Prediction model.
        song_df (pd.DataFrame): DataFrame containing song data.
        context_df (pd.DataFrame): DataFrame containing contextual features.
        user_df (pd.DataFrame): DataFrame containing user features.
        batch_size (int): Batch size for predictions.
        set_batch_size (int, optional): Determines which prediction method to use based on its value:
            - If `True`, use `model.predict` with the specified batch size.
            - If `False`, use `model.predict`.
            - Otherwise, use `model.predict_on_batch`.(defalut)
        save_batch_file (bool, optional): If True, save each batch to a file. Default is False.

        Returns:
        pd.DataFrame: DataFrame of results sorted by prediction score.
        """
        start_time = time.time()
        
        _time_lock = threading.Lock()
        total_preparation_duration = 0
        total_prediction_duration = 0
        num_batches = 0
        
        predictions = []
        all_song_ids = []
        start_values = []
        
        total_size = 359966

        for i in range(0, total_size, batch_size):
            start_values.append(i)
            end = min(i + batch_size - 1, total_size - 1)
            if end == total_size - 1:
                break

        if len(start_values) > 1:
            last_start = start_values.pop()
            start_values.insert(0, last_start)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in start_values:
                futures.append(executor.submit(PredictionProcessor.__preprocess_and_predict_batch_time, 
                                               user_id, song_df, context_df, user_df, model, i, batch_size, set_batch_size, save_batch_file))
            
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
        print(f"{'===== Result preparation took':<38} {result_prep_time - start_time:>6.4f} seconds =====")
        
        result_df = pd.DataFrame({
            'song_id': np.concatenate(all_song_ids),
            'prediction': np.concatenate(predictions)
        })
        result_df = result_df.sort_values(by='prediction', ascending=False)
        
        end_time = time.time()
        print(f"{'===== Total predict_for_user took':<38} {end_time - start_time:>6.4f} seconds =====")
        
        return result_df, average_prep_time, average_predict_time, (result_prep_time - start_time), (end_time - start_time)
    
    @staticmethod
    def preprocess_data_time(user_id, read_from_database=False):
        start_time = time.time()
        if read_from_database:
            context_df, user_df = MySQLDatabase().get_features_by_user_id(user_id)
            read_time = time.time()
            print(f"{'      Querying database took':<38} {read_time - start_time:>6.4f} seconds")
        else:
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
    def __preprocess_and_predict_batch_time(user_id, song_df, context_df, user_df, model, start_index, batch_size, set_batch_size, save_batch_file=False):
        
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
        
        # Choose the prediction method
        predict_method = (model.predict_on_batch if set_batch_size is None else
                          model.predict if set_batch_size is False else
                          lambda x: model.predict(x, batch_size=batch_size))
        
        print(predict_method)
        
        with tf.device('/gpu:0'):
            batch_predictions = predict_method(x_prediction)

        batch_predict_time = time.time()
        prediction_duration = batch_predict_time - batch_prepare_time
        print(f"      {start_index:06} batch prediction took {prediction_duration:>10.4f} seconds")
        
        if save_batch_file:
            FileManage.save_processed_data(user_id_repeated, batch_song_ids, batch_predictions, batch_context_features, batch_user_song_feature, start_index)
        
        # Return the batch song IDs, predictions, preparation time, and prediction time
        return batch_song_ids, batch_predictions, preparation_duration, prediction_duration

class ResultsHandler:
    def __init__(self, model, history, path='.', model_name='model'):
        self.model = model
        self.history = history
        self.path = path
        self.model_name = model_name
        
    def save_results(self, save_architecture=False):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        model_filename = f"{self.model_name}.h5"
        history_filename = f"{self.model_name}_training_history.csv"
        
        model_path = os.path.join(self.path, model_filename)
        history_path = os.path.join(self.path, history_filename)
        
        def get_unique_path(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            directory, filename = os.path.split(file_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{timestamp}{ext}"
            return os.path.join(directory, new_filename)
        
        # Check and handle existing model file
        if os.path.exists(model_path):
            print(f"The file '{model_path}' already exists.")
            user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
            if user_input == 'y':
                pass  # Proceed to overwrite
            else:
                model_path = get_unique_path(model_path)
                print(f"Saving model as '{model_path}' instead.")
        
        # Check and handle existing history file
        if os.path.exists(history_path):
            print(f"The file '{history_path}' already exists.")
            user_input = input("Do you want to overwrite it? (y/n): ").strip().lower()
            if user_input == 'y':
                pass  # Proceed to overwrite
            else:
                history_path = get_unique_path(history_path)
                print(f"Saving training history as '{history_path}' instead.")
                
        # Save the model architecture (optional)
        if save_architecture:
            architecture_path = os.path.join(self.path, f"{self.model_name}_model.png")
            plot_model(self.model, 
                       to_file=architecture_path, 
                       show_shapes=True, 
                       show_layer_names=True, 
                       rankdir="BT", 
                       dpi=200)
            print(f"Model architecture saved to {architecture_path}")
            
        # Save the model
        self.model.save(model_path)
        print(f"Model has been saved as {model_path}")
        
        # Save the training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(history_path, index=False)
        print(f"Training history has been saved as {history_path}")

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
        
        plot_filename = f'{self.model_name}_training_plot.png'
        plot_path = os.path.join(self.path, plot_filename)
        
        plt.savefig(plot_path)
        print(f"Plot has been saved as {plot_path}")
        
        if plot_show:
            plt.show()