from tensorflow.keras.models import load_model
from recommendation_model import PredictionProcessor
from data_management import FileManage
import time

def main():
    model = load_model(r'D:\best_model_v4.h5')
    song_df = FileManage.read_files(file='rec_song')
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
            result_df = PredictionProcessor.predict_for_user(user_id, model, song_df, context_df, user_df, batch_size=120000)
            end_time = time.time()
            print(f"{'===== Total predict_for_user took':<38} {end_time - start_time:>6.4f} seconds =====")
            
            print(f'User {user_id} recommendations:')
            print(len(result_df))
            print(result_df.head(20))
        
        except ValueError:
            print("Invalid user ID. Please enter a numeric user ID.")

if __name__ == "__main__":
    main()