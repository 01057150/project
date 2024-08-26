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
        - Otherwise, use `model.predict_on_batch`.(Defalut)
    save_batch_file (bool, optional): If True, save each batch to a file. Default is False.

    Returns:
    pd.DataFrame: DataFrame of results sorted by prediction score.
"""
from tensorflow.keras.models import load_model
from my_package import PredictionProcessor
from my_package import FileManage
import time

def main():
    model = load_model(r'model/model.h5')
    #song_df = FileManage.read_files(file='rec_song')
    song_df = FileManage.load_from_joblib(r'data/rec_song.joblib')
    model.summary()

    while True:
        # Prompt for user ID input
        user_id_input = input("Enter a user ID (or type 'exit' to quit): ")
        
        if user_id_input.lower() == 'exit':
            print("Exiting the program.")
            break
        
        try:
            # Convert the input to an integer
            user_id = int(user_id_input)
            start_time = time.time()
            context_df, user_df = PredictionProcessor.preprocess_data(user_id,read_from_database=True)
            
            # Make predictions for the user
            result_df = PredictionProcessor.predict_for_user(user_id, model, song_df, context_df, user_df, batch_size=100000)
            end_time = time.time()
            print(f"{'===== Total predict_for_user took':<38} {end_time - start_time:>6.4f} seconds =====")
            
            print(f'User {user_id} recommendations:')
            print(len(result_df))
            print(result_df.head(20))
        
        except ValueError:
            print("Invalid user ID. Please enter a numeric user ID.")

if __name__ == "__main__":
    main()