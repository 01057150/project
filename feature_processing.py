import os
import pickle
import numpy as np
import pandas as pd
from joblib import dump, load
from data_management import FileManage
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer, MultiLabelBinarizer, OneHotEncoder

class FeatureAdder:
    # Function: Add number_of_songs
    @staticmethod
    def user_features(merged_df):
        # Check if msno or user_id column exists
        if 'msno' in merged_df.columns:
            user_column = 'msno'
        elif 'user_id' in merged_df.columns:
            user_column = 'user_id'
        else:
            raise ValueError("The DataFrame must contain either 'msno' or 'user_id' column.")
        
        # Add number_of_songs column, counting occurrences of each user
        merged_df['number_of_songs'] = merged_df.groupby(user_column)[user_column].transform('count')
        
        print("Number of songs per user added.")
        
        return merged_df

    # Function: Add support, confidence, artist_support, artist_confidence
    @staticmethod
    def song_features(merged_df, context_feature):
        # Ensure 'song_id' and 'target' columns are present in context_feature
        if 'song_id' not in context_feature.columns or 'target' not in context_feature.columns:
            raise ValueError("The 'context_feature' DataFrame must contain 'song_id' and 'target' columns.")
        
        # Calculate support and confidence
        song_counts = context_feature['song_id'].value_counts(normalize=True)
        song_target_counts = context_feature[context_feature['target'] == 1]['song_id'].value_counts()
        song_total_counts = context_feature['song_id'].value_counts()

        # Add support
        merged_df = FeatureAdder.__calculate_support(merged_df, song_counts)

        # Add confidence
        merged_df = FeatureAdder.__calculate_confidence(merged_df, song_target_counts, song_total_counts)

        print("Song support and confidence metrics added.")
        
        # Fill NaN values in artist_name
        merged_df.fillna({'artist_name': 'Unknown'}, inplace=True)

        # Ensure 'artist_name' column is present in merged_df
        if 'artist_name' not in merged_df.columns:
            raise ValueError("The DataFrame must contain 'artist_name' column.")
        
        # Calculate support and confidence for each artist
        artist_counts = merged_df.groupby('artist_name').size()
        artist_target_counts = merged_df[merged_df['target'] == 1].groupby('artist_name').size() 

        # Add artist support
        merged_df = FeatureAdder.__calculate_artist_support(merged_df, artist_counts)

        # Add artist confidence
        merged_df = FeatureAdder.__calculate_artist_confidence(merged_df, artist_target_counts, artist_counts)

        print("Artist support and confidence metrics added.")
        
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
    
    @staticmethod
    def user_features(merged_df, mode='train', scaler_dir='Scalers', scaler_filename='user_scaler.joblib'):
        scaler_path = NumericProcessor.__ensure_directory(scaler_dir, scaler_filename)
        numeric_user_cols = ['number_of_songs', 'bd']
        
        # Log transform for skewed data
        merged_df = NumericProcessor.__log_transform(merged_df, numeric_user_cols)
        
        if mode == 'train':
            scaler = StandardScaler()
            merged_df[numeric_user_cols] = scaler.fit_transform(merged_df[numeric_user_cols])
            dump(scaler, scaler_path)
            print(f"StandardScaler for user features saved to {scaler_path}.")
        elif mode == 'prediction':
            merged_df = NumericProcessor.__load_and_transform(merged_df, numeric_user_cols, scaler_path)
        else:
            raise ValueError("Mode should be 'train' or 'prediction'.")
        
        print("User numeric features standardized.")
        return merged_df
    
    @staticmethod
    def song_features(merged_df, mode='train', scaler_dir='Scalers', scaler_filename='song_scaler.joblib'):
        scaler_path = NumericProcessor.__ensure_directory(scaler_dir, scaler_filename)
        numeric_song_cols = ['support', 'confidence', 'artist_support', 'artist_confidence']
        
        # Log transform for specific columns
        merged_df = NumericProcessor.__log_transform(merged_df, ['support', 'artist_support'])
        
        if mode == 'train':
            scaler = StandardScaler()
            merged_df[numeric_song_cols] = scaler.fit_transform(merged_df[numeric_song_cols])
            dump(scaler, scaler_path)
            print(f"StandardScaler for user features saved to {scaler_path}.")
        elif mode == 'prediction':
            merged_df = NumericProcessor.__load_and_transform(merged_df, numeric_song_cols, scaler_path)
        else:
            raise ValueError("Mode should be 'train' or 'prediction'.")
        
        print("Song numeric features standardized.")
        return merged_df
    
    @staticmethod
    def version1(merged_df):
        max_number_of_songs = merged_df['number_of_songs'].max()
        merged_df['number_of_songs'] = merged_df['number_of_songs'] / max_number_of_songs
        print("number_of_songs has been scaled to 0~1 range")
        
        normalizer = Normalizer()
        merged_df['bd'] = normalizer.fit_transform(merged_df[['bd']])
        print("bd has been normalized")
        
        return merged_df
    
    @staticmethod
    def version2(merged_df):
        numeric_cols = ['number_of_songs', 'support', 'confidence', 'artist_support', 'artist_confidence', 'bd']
        merged_df = NumericProcessor.version1(merged_df)
        
        scaler = StandardScaler()
        merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])
        print("All numeric features have been standardized")
        
        return merged_df
    
    # Private method: Log transformation
    @staticmethod
    def __log_transform(df, cols):
        for col in cols:
            df[col] = np.log1p(df[col])
            print(f"Log-transformed {col}")
        return df

    # Private method: Ensure directory and scaler path
    @staticmethod
    def __ensure_directory(scaler_dir, scaler_filename):
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        return os.path.join(scaler_dir, scaler_filename)
    
    # Private method: Load scaler and transform data
    @staticmethod
    def __load_and_transform(df, cols, scaler_path):
        try:
            scaler = load(scaler_path)
            print("StandardScaler loaded successfully")
            df[cols] = scaler.transform(df[cols])
        except FileNotFoundError:
            raise FileNotFoundError(f"StandardScaler {scaler_path} not found: ")
        return df
    

class ContextualProcessor:
    # Function: Process contextual features (original paper method)
    @staticmethod
    def clean_contextual_features(merged_df):
        # Define valid values for contextual features, note that source_screen_name starts with a capital letter
        valid_source_system_tab = ['my library', 'discover', 'radio', 'listen with', 'others']
        valid_source_screen_name = ['Local playlist more', 'Online playlist more', 'My library', 'Radio', 'Others profile more', 'Discover Genre', 'Others']
        valid_source_type = ['online-playlist', 'local-playlist', 'local-library', 'radio', 'listen-with', 'others']
        
        # Replace invalid values with 'others'
        merged_df['source_system_tab'] = merged_df['source_system_tab'].apply(lambda x: x if x in valid_source_system_tab else 'others')
        merged_df['source_screen_name'] = merged_df['source_screen_name'].apply(lambda x: x if x in valid_source_screen_name else 'Others')
        merged_df['source_type'] = merged_df['source_type'].apply(lambda x: x if x in valid_source_type else 'others')

        print("Invalid values in contextual features processed.")
        
        return merged_df
    
    @staticmethod
    def clean_contextual_features_simplify(merged_df):
        # Define valid values for contextual features, note that source_screen_name starts with a capital letter
        valid_source_system_tab = ['my library', 'discover', 'others']
        valid_source_screen_name = ['Playlist more', 'My library', 'Discover Genre', 'Others']
        valid_source_type = ['online-playlist', 'local-playlist', 'local-library', 'others']
        
        # Replace invalid values with 'others'
        merged_df['source_system_tab'] = merged_df['source_system_tab'].apply(lambda x: x if x in valid_source_system_tab else 'others')
        
        # Change 'Local playlist more' and 'Online playlist more' to 'Playlist more'
        merged_df['source_screen_name'] = merged_df['source_screen_name'].apply(lambda x: 'Playlist more' if x in ['Local playlist more', 'Online playlist more'] else (x if x in valid_source_screen_name else 'Others'))
        
        # Change 'online-playlist' and 'local-playlist' to 'playlist'
        merged_df['source_type'] = merged_df['source_type'].apply(lambda x: 'playlist' if x in ['online-playlist', 'local-playlist'] else (x if x in valid_source_type else 'others'))

        print("Invalid values in contextual_features have been processed")
        
        return merged_df


class FeatureProcessor:
    # Function: Process genre_ids, noting that multiple genres are separated by '|'
    @staticmethod
    def genre_ids(merged_df):
        # Define valid genre IDs
        valid_genre_ids = ['465', '458', '921', '1609', '444', '1259', '2022', '359', '139', '2122', 'others']
        
        def replace_invalid_genres(genre_str):
            genres = genre_str.split('|')
            replaced_genres = [genre if genre in valid_genre_ids else 'others' for genre in genres]
            return replaced_genres
        
        # Replace invalid values with 'others' and process the genre_ids column as a list
        merged_df['genre_ids'] = merged_df['genre_ids'].astype(str).apply(replace_invalid_genres)
        
        print("Invalid values in genre IDs features processed.")
        
        # Encode using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(merged_df['genre_ids'])
        
        # Convert the encoded results to a DataFrame
        genre_encoded_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
        
        # Merge the original DataFrame with the encoded genre_ids
        merged_df = pd.concat([merged_df, genre_encoded_df], axis=1)
        
        # Remove the original genre_ids column
        merged_df.drop(columns=['genre_ids'], inplace=True)
        
        print("OneHotEncoder for genre IDs completed.")
        
        return merged_df, mlb.classes_
    
    # Function: Process gender and age
    @staticmethod
    def gender_bd_features(user_feature_df):
        # Fill missing values in the gender column
        user_feature_df.fillna({'gender': 'none'}, inplace=True)

        # Check and replace bd (birthdate/age)
        # Calculate the average age between 14 and 68
        valid_bd = user_feature_df['bd'][(user_feature_df['bd'] >= 14) & (user_feature_df['bd'] <= 68)]
        average_bd = valid_bd.mean()

        # Replace bd values that are not in the 14 to 68 range
        user_feature_df['bd'] = user_feature_df['bd'].apply(lambda x: average_bd if x < 14 or x > 68 else x)

        print("Gender and birthdate modifications completed.")
        
        return user_feature_df
    
    @staticmethod
    def one_hot_encoder(train_df, mode='train'):
        user_ids = train_df['user_id'].values
        song_ids = train_df['song_id'].values
            
        genre_columns = ['465', '458', '921', '1609', '444', '1259', '2022', '359', '139', '2122', 'others']
        genre_features_encoded = train_df[genre_columns].values

        contextual_features_encoded = FeatureProcessor.one_hot_encode_contextual_features(
            train_df[['source_system_tab', 'source_screen_name', 'source_type']], mode=mode
        )

        print("First few rows of merged contextual_features:")
        print(pd.DataFrame(contextual_features_encoded).head())
            
        gender_features_encoded = FeatureProcessor.one_hot_encode_gender_features(train_df[['gender']], mode=mode)

        numeric_song_features = train_df[['support', 'confidence', 'artist_support', 'artist_confidence']].values
        numeric_user_features = train_df[['number_of_songs', 'bd']].values
        
        user_song_features = np.concatenate([
            numeric_user_features,
            gender_features_encoded,
            genre_features_encoded,
            numeric_song_features
        ], axis=1)

        print("First few rows of merged user_song_features:")
        print(pd.DataFrame(user_song_features).head())

        targets = train_df['target'].values

        x_train = [user_ids, song_ids, contextual_features_encoded, user_song_features]
        y_train = targets
            
        print('Data preparation is complete')
        return x_train, y_train

    @staticmethod
    def __get_or_create_encoder(df, mode, encoder_dir, encoder_filename):
        if not os.path.exists(encoder_dir):
            os.makedirs(encoder_dir)

        encoder_path = os.path.join(encoder_dir, encoder_filename)
        
        if mode == 'train':
            encoder = OneHotEncoder(sparse_output=False)
            encoded_features = encoder.fit_transform(df)
            dump(encoder, encoder_path)
            print(f"Encoder saved to {encoder_path}")
        elif mode in ['prediction', 'test']:
            if os.path.exists(encoder_path):
                encoder = load(encoder_path)
                encoded_features = encoder.transform(df)
                print(f"Encoder loaded from {encoder_path}")
            else:
                raise FileNotFoundError(f"Encoder {encoder_filename} not found. Please ensure the encoder was saved during the training phase.")
        else:
            raise ValueError("Mode should be 'train', 'prediction', or 'test'")
        
        return encoded_features

    # Function: One-Hot encode contextual features
    @staticmethod
    def one_hot_encode_contextual_features(df, mode='train', encoder_dir='OneHotEncoder', encoder_filename='contextual_encoder.joblib'):
        contextual_features = df[['source_system_tab', 'source_screen_name', 'source_type']]
        return FeatureProcessor.__get_or_create_encoder(contextual_features, mode, encoder_dir, encoder_filename)

    # Function: One-Hot encode gender features
    @staticmethod
    def one_hot_encode_gender_features(df, mode='train', encoder_dir='OneHotEncoder', encoder_filename='gender_encoder.joblib'):
        gender_features = df[['gender']]
        return FeatureProcessor.__get_or_create_encoder(gender_features, mode, encoder_dir, encoder_filename)


class Encoder:
    # Function: Encode msno, song_id
    @staticmethod
    def encode(merged_df, path):
        # Ensure the directory exists
        Encoder.__check_and_create_path(path)

        # Create LabelEncoder instances
        msno_encoder = LabelEncoder()
        song_id_encoder = LabelEncoder()
        merged_df['msno_encoded'] = msno_encoder.fit_transform(merged_df['msno'])
        merged_df['song_id_encoded'] = song_id_encoder.fit_transform(merged_df['song_id'])
        
        # Save mappings
        msno_mapping = pd.DataFrame({
            'msno': msno_encoder.classes_,
            'msno_encoded': range(len(msno_encoder.classes_))
        })
        
        song_id_mapping = pd.DataFrame({
            'song_id': song_id_encoder.classes_,
            'song_id_encoded': range(len(song_id_encoder.classes_))
        })
        
        # Save LabelEncoders
        Encoder.save_label_encoders(msno_encoder, song_id_encoder, path)
        
        # Calculate dimensions
        #num_users = msno_mapping.max()
        #num_songs = song_id_mapping.max()
        num_users = len(msno_mapping)
        num_songs = len(song_id_mapping)
        print(f"Number of users: {num_users}.")
        print(f"Number of songs: {num_songs}.")
        
        # Save to CSV files
        FileManage.save_to_csv(msno_mapping, os.path.join(path, 'msno_mapping.csv'))
        FileManage.save_to_csv(song_id_mapping, os.path.join(path, 'song_id_mapping.csv'))

        return merged_df, num_users, num_songs
    
    # Save LabelEncoder instances
    @staticmethod
    def save_label_encoders(msno_encoder, song_id_encoder, path):
        Encoder.__check_and_create_path(path)
        with open(os.path.join(path, 'msno_encoder.pkl'), 'wb') as f:
            pickle.dump(msno_encoder, f)
        with open(os.path.join(path, 'song_id_encoder.pkl'), 'wb') as f:
            pickle.dump(song_id_encoder, f)

    # Load LabelEncoder instances
    @staticmethod
    def load_label_encoders(path):
        with open(os.path.join(path, 'msno_encoder.pkl'), 'rb') as f:
            msno_encoder = pickle.load(f)
        with open(os.path.join(path, 'song_id_encoder.pkl'), 'rb') as f:
            song_id_encoder = pickle.load(f)
        return msno_encoder, song_id_encoder

    # Update LabelEncoder with new values
    @staticmethod
    def update_label_encoders(encoder, new_values):
        new_classes = pd.Series(new_values).unique()
        current_classes = encoder.classes_
        updated_classes = pd.Series(current_classes).append(pd.Series(new_classes)).unique()
        encoder.classes_ = updated_classes
        return encoder
    
    # Check and create path if it doesn't exist
    @staticmethod
    def __check_and_create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)