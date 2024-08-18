import gc
import os
import json
import joblib
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split

class MySQLDatabase:
    def __init__(self, config_path='config.json'):
        """
        Initialize the database connection.
        :param config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish a database connection and create a cursor object."""
        try:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)
            self.connection = mysql.connector.connect(**config)
            self.cursor = self.connection.cursor()
            print("Database connection successful!")
        except mysql.connector.Error as err:
            print(f"Database connection failed: {err}")
            self.connection = None
            self.cursor = None

    def execute_query(self, query, params=None):
        """
        Execute an SQL query.
        :param query: SQL query string.
        :param params: Query parameters (optional).
        :return: Query result.
        """
        if self.connection and self.cursor:
            try:
                self.cursor.execute(query, params)
                return self.cursor.fetchall()
            except mysql.connector.Error as err:
                print(f"Error during execution: {err}")
                return None
        else:
            print("Not connected to the database.")
            return None
        
    def get_features_by_user_id(self, user_id):
        """
        Retrieve features from the database based on user ID.
        :param user_id: The user ID to retrieve features for.
        :return: Two DataFrames containing context and user features.
        """
        self.connect()  # Establish the database connection
        if user_id is not None:
            user_id = int(user_id)  # Ensure user_id is an integer
            
            # Query context features
            query_context = "SELECT song_id, source_system_tab, source_screen_name, source_type FROM c_features WHERE user_id=%s"
            context_features = self.execute_query(query_context, (user_id,))

            # Query user features
            query_user = "SELECT number_of_songs, bd, gender FROM u_features WHERE user_id=%s"
            user_features = self.execute_query(query_user, (user_id,))
            
            self.close()  # Close the database connection
            
            context_feature_df = pd.DataFrame(context_features, columns=['song_id', 'source_system_tab', 'source_screen_name', 'source_type'])
            user_feature_df = pd.DataFrame(user_features, columns=['number_of_songs', 'bd', 'gender'])
            
            # Clean up any unwanted characters from the DataFrame
            user_feature_df['gender'] = user_feature_df['gender'].str.strip()
            
            return context_feature_df, user_feature_df
        
        self.close()  # Close the database connection even if user_id is None
        return pd.DataFrame(), pd.DataFrame()

    def close(self):
        """Close the cursor and database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Database connection closed.")

class FileManage:
    # Initialize file paths
    _paths = None

    @classmethod
    def initialize_paths(cls):
        """Initialize file paths from the file_paths.json."""
        cls._paths = cls.load_paths()

    @staticmethod
    def load_paths():
        """Load file paths from file_paths.json."""
        try:
            with open('file_paths.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("file_paths.json not found. Please make sure this file exists.")
            raise
        except json.JSONDecodeError:
            print("Error decoding file_paths.json. Please ensure it's properly formatted.")
            raise

    @classmethod
    def get_paths(cls):
        """Get the file paths after initializing."""
        if cls._paths is None:
            cls.initialize_paths()
        return cls._paths

    @staticmethod
    def read_raw_files():
        """Read raw data files."""
        paths = FileManage.get_paths()
        context_feature_df = pd.read_csv(paths['train'], encoding='utf-8')
        song_feature_df = pd.read_csv(paths['songs'], encoding='utf-8')
        user_feature_df = pd.read_csv(paths['members'], encoding='utf-8')

        print("Files read successfully.")

        return (context_feature_df, song_feature_df[['song_id', 'genre_ids', 'artist_name']], user_feature_df[['msno', 'bd', 'gender']])

    @staticmethod
    def read_files(user_id=None, file='rec_song'):
        """Read various files based on the provided parameters."""
        paths = FileManage.get_paths()
        try:
            if user_id is not None:
                context_feature_df = pd.read_csv(paths['context_data'])
                user_feature_df = pd.read_csv(paths['user_data'])

                print("Files read successfully.")

                # Set index to speed up queries
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

                # Reset index for further operations
                context_feature_df_filtered.reset_index(inplace=True)
                user_feature_df_filtered.reset_index(inplace=True)

                return (context_feature_df_filtered[['song_id', 'source_system_tab', 'source_screen_name', 'source_type']],
                        user_feature_df_filtered)

            # Handle file reading based on the 'file' parameter
            if file in paths:
                df = pd.read_csv(paths[file])
                return df
            else:
                raise ValueError(f"Invalid file parameter: {file}")

        except FileNotFoundError:
            print(f"File '{file}' not found, please check the path.")
        except KeyError:
            print(f"Key '{file}' not found in file_paths.json.")

    @staticmethod
    def save_processed_data(user_id_repeated, batch_song_ids, batch_predictions, batch_context_features, batch_user_song_feature, index):
        """Save processed data to CSV."""
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
        file_path = os.path.join(FileManage.get_paths()['save_batch_file'], file_name)

        result_df.to_csv(file_path, index=False)

        print(f'Saved data to {file_path}')
    
    @staticmethod
    def save_to_csv(df, file_path, precision=8):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created")
        
        if os.path.exists(file_path):
            # Sort data frame to ensure consistent order
            df_sorted = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
            df_sorted = df_sorted.round(precision)  # Round to the specified precision
            
            # Read existing file
            existing_df = pd.read_csv(file_path)
            existing_df_sorted = existing_df.sort_values(by=existing_df.columns.tolist()).reset_index(drop=True)
            existing_df_sorted = existing_df_sorted.round(precision)  # Round to the specified precision

            # Compare file contents, including floating point precision
            try:
                pd.testing.assert_frame_equal(df_sorted, existing_df_sorted, check_dtype=False, check_exact=True)
                print(f"File {file_path} exists and is identical to the source.")
                return
            except AssertionError:
                # Show warning and compare differences
                print(f"Warning: File {file_path} exists but contents differ. Overwriting the file.")
                
                differences = df_sorted.compare(existing_df_sorted)
                print("The following differences were found:")
                print(differences)
                
                # Prompt user for confirmation
                user_input = input("Do you want to overwrite the file? Enter 'y' to confirm, any other key to cancel: ")
                if user_input.lower() != 'y':
                    print("Operation cancelled.")
                    return
        else:
            # File does not exist, create new one
            print(f"File {file_path} does not exist, creating new file.")

        # Save new CSV file
        df.to_csv(file_path, index=False)
        print(f"Results saved to: {file_path}")
        
    @staticmethod
    def save_to_joblib(df, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created")

        if os.path.exists(file_path):
            # Load existing DataFrame from joblib
            existing_df = joblib.load(file_path)

            # Sort DataFrames for comparison
            df_sorted = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)
            existing_df_sorted = existing_df.sort_values(by=existing_df.columns.tolist()).reset_index(drop=True)

            # Compare DataFrames
            try:
                pd.testing.assert_frame_equal(df_sorted, existing_df_sorted, check_dtype=False, check_exact=True)
                print(f"File {file_path} exists and is identical to the source.")
                return
            except AssertionError:
                print(f"Warning: File {file_path} exists but contents differ. Overwriting the file.")
                
                differences = df_sorted.compare(existing_df_sorted)
                print("The following differences were found:")
                print(differences)
                
                user_input = input("Do you want to overwrite the file? Enter 'y' to confirm, any other key to cancel: ")
                if user_input.lower() != 'y':
                    print("Operation cancelled.")
                    return
        else:
            print(f"File {file_path} does not exist, creating new file.")

        # Save new DataFrame to joblib
        joblib.dump(df, file_path)
        print(f"Results saved to: {file_path}")
        
    @staticmethod    
    def load_from_joblib(file_path):
        if os.path.exists(file_path):
            return joblib.load(file_path)
        else:
            print(f"File {file_path} does not exist.")
            return None
        
class FileComparer:
    @staticmethod
    def compare_song_ids(original_file, context_file, columns_to_check):
        """
        Compare song IDs between an original file and a context file.
        :param original_file: Path to the original file.
        :param context_file: Path to the context file.
        :param columns_to_check: Columns to check for non-zero values.
        """
        # Read data from files
        original_df = pd.read_csv(original_file)
        context_df = pd.read_csv(context_file)

        # Filter rows where the sum of specified columns is not zero
        filtered_df = original_df[original_df[columns_to_check].sum(axis=1) != 0]

        # Extract unique song IDs and user IDs from the filtered data
        filtered_song_ids = filtered_df['song_id'].unique()
        filtered_user_ids = filtered_df['user_id'].unique()

        # Find matching songs in the context data
        matching_songs = context_df[context_df['user_id'].isin(filtered_user_ids)]
        context_song_ids = matching_songs['song_id'].unique()

        # Determine common, unique filtered, and unique context song IDs
        common_song_ids = set(filtered_song_ids).intersection(set(context_song_ids))
        unique_filtered_song_ids = set(filtered_song_ids) - common_song_ids
        unique_context_song_ids = set(context_song_ids) - common_song_ids

        # Output results
        print(f"Number of common song_id in both sets: {len(common_song_ids)}.")

        print("\nSong IDs in the original data but not in context_data.csv:")
        for song_id in unique_filtered_song_ids:
            print(song_id)

        print("\nSong IDs in context_data.csv but not in the original data:")
        for song_id in unique_context_song_ids:
            print(song_id)

    @staticmethod
    def compare_file(df_or_path1, df_or_path2, compare_columns, precision=6):
        """
        Compare the contents of two DataFrames or files based on specific columns.
        :param df_or_path1: DataFrame or file path for the first set of data.
        :param df_or_path2: DataFrame or file path for the second set of data.
        :param compare_columns: Columns to compare.
        :param precision: Number of decimal places for comparison.
        """
        # Helper function to load DataFrame from file path if needed
        def load_dataframe(df_or_path):
            if isinstance(df_or_path, pd.DataFrame):
                return df_or_path
            elif isinstance(df_or_path, str) and os.path.exists(df_or_path):
                return pd.read_csv(df_or_path)
            else:
                raise ValueError("Invalid input: must be a DataFrame or a valid file path.")

        # Load DataFrames
        df1 = load_dataframe(df_or_path1)
        df2 = load_dataframe(df_or_path2)

        # Ensure columns to compare exist in the DataFrames
        missing_cols_df1 = [col for col in compare_columns if col not in df1.columns]
        missing_cols_df2 = [col for col in compare_columns if col not in df2.columns]

        if missing_cols_df1:
            raise ValueError(f"First DataFrame/File is missing the following columns: {', '.join(missing_cols_df1)}.")
        if missing_cols_df2:
            raise ValueError(f"Second DataFrame/File is missing the following columns: {', '.join(missing_cols_df2)}.")

        # Select columns to compare
        df1_to_compare = df1[compare_columns]
        df2_to_compare = df2[compare_columns]

        # Set precision for comparison
        pd.set_option('display.precision', precision)

        # Round values to the specified precision
        df1_to_compare = df1_to_compare.round(precision)
        df2_to_compare = df2_to_compare.round(precision)

        # Display DataFrames for verification
        print("First DataFrame/File to compare:")
        print(df1_to_compare)
        print("\nSecond DataFrame/File to compare:")
        print(df2_to_compare)

        # Compare DataFrames
        try:
            pd.testing.assert_frame_equal(df1_to_compare, df2_to_compare, check_dtype=False, check_exact=False)
            print("The contents of the two DataFrames/Files are the same.")
        except AssertionError:
            # Show warning and differences
            print("Warning: The contents of the two DataFrames/Files are different.")

            # Display specific differences
            differences = df1_to_compare.compare(df2_to_compare)
            print("Differences between the two DataFrames/Files:")
            print(differences)
        
        
class DataHandler:
    @staticmethod
    def merge_raw_data(context_feature_df, song_feature_df, user_feature_df):
        """
        Merge raw data from context, song, and user feature dataframes.

        :param context_feature_df: DataFrame containing context features.
        :param song_feature_df: DataFrame containing song features.
        :param user_feature_df: DataFrame containing user features.
        :return: Merged DataFrame.
        """
        # Ensure that necessary columns are present
        required_context_cols = ['song_id']
        required_song_cols = ['song_id']
        required_user_cols = ['msno']
        
        missing_context_cols = [col for col in required_context_cols if col not in context_feature_df.columns]
        missing_song_cols = [col for col in required_song_cols if col not in song_feature_df.columns]
        missing_user_cols = [col for col in required_user_cols if col not in user_feature_df.columns]
        
        if missing_context_cols:
            raise ValueError(f"Missing columns in context_feature_df: {missing_context_cols}.")
        if missing_song_cols:
            raise ValueError(f"Missing columns in song_feature_df: {missing_song_cols}.")
        if missing_user_cols:
            raise ValueError(f"Missing columns in user_feature_df: {missing_user_cols}.")

        # Merge dataframes
        merged_df = pd.merge(context_feature_df, song_feature_df, on='song_id', how='left')
        merged_df = pd.merge(merged_df, user_feature_df, left_on='msno', right_on='msno', how='left')

        print("Data merge completed successfully.")
        
        return merged_df
    
    @staticmethod
    def drop_rename_col(merged_df, exclude_cols=None, rename_cols=None):
        """
        Drop and rename columns in the DataFrame.

        :param merged_df: DataFrame to process.
        :param exclude_cols: List of columns to drop.
        :param rename_cols: Dictionary mapping old column names to new names.
        :return: Modified DataFrame.
        """
        # Set default values
        if exclude_cols is None:
            exclude_cols = ['msno', 'song_id']
        if rename_cols is None:
            rename_cols = {'msno_encoded': 'user_id', 'song_id_encoded': 'song_id'}

        # Drop specified columns
        if not set(exclude_cols).issubset(merged_df.columns):
            missing_cols = list(set(exclude_cols) - set(merged_df.columns))
            raise ValueError(f"Columns missing from DataFrame: {missing_cols}")
        
        merged_df.drop(columns=exclude_cols, inplace=True)
        print(f"Dropped columns: {exclude_cols}.")

        # Rename specified columns
        if not set(rename_cols.keys()).issubset(merged_df.columns):
            missing_rename_cols = list(set(rename_cols.keys()) - set(merged_df.columns))
            raise ValueError(f"Cannot rename columns, missing columns: {missing_rename_cols}.")
        
        merged_df.rename(columns=rename_cols, inplace=True)
        print(f"Renamed columns: {rename_cols}.")
        
        return merged_df
    
    @staticmethod
    def reorder_columns(merged_df, genre_classes=None):
        """
        Reorder columns in the DataFrame.

        :param merged_df: DataFrame to process.
        :param genre_classes: List of genre columns to include.
        :return: DataFrame with reordered columns.
        """
        if genre_classes is not None:
            genre_columns = list(genre_classes)  # Include encoded genre_ids
            
            new_order = [
                'user_id', 'song_id', 'target',
                'source_system_tab', 'source_screen_name', 'source_type',
                'number_of_songs', 'bd', 'gender'
            ] + genre_columns + [
                'support', 'confidence', 'artist_support', 'artist_confidence'
            ]
        else:
            new_order = [
                'user_id', 'song_id',
                'source_system_tab', 'source_screen_name', 'source_type',
                'number_of_songs', 'bd', 'gender'
            ]
        
        # Check for columns missing from DataFrame
        missing_columns = [col for col in new_order if col not in merged_df.columns]
        if missing_columns:
            print(f"Columns missing from DataFrame, will skip: {missing_columns}")

        # Reorder columns based on what exists
        new_order = [col for col in new_order if col in merged_df.columns]
        merged_df = merged_df.reindex(columns=new_order)
        
        print("Columns reordered.")
        
        return merged_df
    
    @staticmethod
    def release_memory(*dfs):
        """
        Release memory by deleting DataFrames and invoking garbage collection.

        :param dfs: DataFrames to be released.
        """
        for df in dfs:
            del df
        gc.collect()
        print("Memory has been released.")

class Recommender:
    @staticmethod
    def rec_song(merged_df, path, num_songs, genre_classes, save_format='csv'):
        rec_song_path = os.path.join(path, f'rec_song.{save_format}')
        
        genre_columns = list(genre_classes)
        # Retain the required columns
        retained_columns = [
            'song_id_encoded'] + genre_columns + [
            'support', 'confidence', 'artist_support', 'artist_confidence'
        ]
        filtered_df = merged_df.loc[:, retained_columns]
        
        # Remove duplicates
        filtered_df = filtered_df.drop_duplicates(subset=['song_id_encoded'])
        
        # Sort by song_id_encoded
        filtered_df.sort_values(by='song_id_encoded', inplace=True)
        
        # Check for any missing song_id_encoded
        all_song_ids = set(range(1, num_songs + 1))  # Changed to num_songs + 1 to include all possible IDs
        present_song_ids = set(filtered_df['song_id_encoded'])
        missing_song_ids = all_song_ids - present_song_ids
        if missing_song_ids:
            print(f"Missing song_id_encoded: {missing_song_ids}")
        else:
            print("All song_id_encoded entries are present.")
            
        filtered_df.rename(columns={'song_id_encoded': 'song_id'}, inplace=True)
        
        # Save the result to the chosen file format
        if save_format == 'csv':
            FileManage.save_to_csv(filtered_df, rec_song_path)
        elif save_format == 'joblib':
            FileManage.save_to_joblib(filtered_df, rec_song_path)
        else:
            print(f"Unsupported file format: {save_format}. Please use 'csv' or 'joblib'.")
        
class DataSplitter:
    @staticmethod
    def split_data(merged_df, path, train_size=0.8, save_format='csv'):
        train_path = os.path.join(path, f'train_file.{save_format}')
        test_path = os.path.join(path, f'test_file_filtered.{save_format}')
        
        # Split the dataset into train and test with the specified ratio
        train_df, test_df = train_test_split(merged_df, train_size=train_size, random_state=42)

        # Filter test set to ensure users and songs exist in the training set
        train_users = set(train_df['user_id'])
        train_items = set(train_df['song_id'])

        test_df = test_df[test_df['user_id'].isin(train_users) & test_df['song_id'].isin(train_items)]

        # Save the processed train and test datasets to new files
        if save_format == 'csv':
            FileManage.save_to_csv(train_df, train_path)
            FileManage.save_to_csv(test_df, test_path)
        elif save_format == 'joblib':
            FileManage.save_to_joblib(train_df, train_path)
            FileManage.save_to_joblib(test_df, test_path)
        else:
            print(f"Unsupported file format: {save_format}. Please use 'csv' or 'joblib'.")

        return train_df