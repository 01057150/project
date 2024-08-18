import os
import numpy as np
from tensorflow.keras.models import load_model
from recommendation_model import PredictionProcessor
from data_management import FileManage

# Define the paths
model_path = r'model/model.h5'  # Path to the saved model
output_dir = r'D:\prediction_results'  # Directory where prediction results will be saved

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set random seed and generate 380 unique random numbers as user IDs
np.random.seed(42)
user_ids = np.random.choice(range(30755), size=380, replace=False)

# Load the model
model = load_model(model_path)

# Load the song data
song_df = FileManage.read_files(file='rec_song')

# Print the model summary
model.summary()

for user_id in user_ids:
    for read_from_database in [True, False]:
        print(f"Running prediction for user {user_id} with read_from_database = {read_from_database}")
        batch_size = 120000
        
        context_df, user_df = PredictionProcessor.preprocess_data(user_id, read_from_database=read_from_database)

        result_df = PredictionProcessor.predict_for_user(user_id, model, song_df, context_df, user_df, batch_size)

        # Determine the file name based on read_from_database flag
        if read_from_database:
            file_path = os.path.join(output_dir, f'user_{user_id}_recommendations_mysql.csv')
        else:
            file_path = os.path.join(output_dir, f'user_{user_id}_recommendations_file.csv')
            
        result_df.to_csv(file_path, index=False)
        print(f'Results for user {user_id} saved to {file_path}')
