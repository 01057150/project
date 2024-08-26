from my_package import FileManage, DataHandler, Recommender, DataSplitter
from my_package import FeatureAdder, NumericProcessor, ContextualProcessor, FeatureProcessor, Encoder
from my_package import MLPModelBuilder, ModelTrainer, ResultsHandler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import os

class RecommendationModel:
    def __init__(self, config):
        self.config = config
        self.path = config['file_path']
        self.model = None
        self.history = None
        self.num_users = 30755
        self.num_songs = 359966
        self.genre_columns = []

    def read_and_process_data(self):
        # Step 1: Read files(database)
        context_feature_df, song_feature_df, user_feature_df = FileManage.read_raw_files()
        
        # Step 2: Process gender and birthdate
        user_feature_df = FeatureProcessor.gender_bd_features(user_feature_df)
        
        # Step 3: Merge data
        merged_df = DataHandler.merge_raw_data(context_feature_df, song_feature_df, user_feature_df)
        
        # Step 4: Add features
        merged_df = FeatureAdder.user_features(merged_df)
        merged_df = FeatureAdder.song_features(merged_df, context_feature_df)
        
        # Step 5: Release memory
        DataHandler.release_memory(context_feature_df, song_feature_df, user_feature_df)
        
        # Step 6: Process numeric and contextual features
        merged_df = NumericProcessor.user_features(merged_df)
        merged_df = NumericProcessor.song_features(merged_df)
        merged_df = ContextualProcessor.clean_contextual_features(merged_df)
        
        # Step 7: Process genre IDs
        merged_df, self.genre_columns = FeatureProcessor.genre_ids(merged_df)
        
        # Step 8: Encode features
        merged_df, self.num_users, self.num_songs = Encoder.encode(merged_df, self.path)
        
        # Step 9: Create rec_song.csv
        Recommender.rec_song(merged_df, self.path, self.num_songs, self.genre_columns, save_format='joblib')
        
        # Step 10: Manage columns
        merged_df = DataHandler.drop_rename_col(merged_df)
        merged_df = DataHandler.reorder_columns(merged_df, self.genre_columns)
        
        return merged_df

    def split_and_prepare_data(self, merged_df):
        # Step 11: Split data
        train_df = DataSplitter.split_data(merged_df, self.path, save_format='joblib')
        
        # Step 12: Prepare data for training
        x_train, y_train = FeatureProcessor.one_hot_encoder(train_df)
        
        return x_train, y_train
    
    def file_based_data_preparation(self):
        # Step 11: Read train files
        #train_df = FileManage.read_files(file = 'train_file')
        train_df_path = os.path.join(self.path, 'train_file.joblib')
        train_df = FileManage.load_from_joblib(train_df_path)
        
        # Step 12: Prepare data for training
        x_train, y_train = FeatureProcessor.one_hot_encoder(train_df)
        
        return x_train, y_train

    def build_model(self):
        # Step 13: Create MLP model
        model_builder = MLPModelBuilder(num_users=self.num_users + 1000,
                                        num_songs=self.num_songs, 
                                        user_feature_dim=self.config['user_feature_dim'], 
                                        song_feature_dim=self.config['song_feature_dim'], 
                                        contextual_feature_dim=self.config['contextual_feature_dim'], 
                                        user_embedding_dim=self.config['user_embedding_dim'], 
                                        song_embedding_dim=self.config['song_embedding_dim'])
        self.model = model_builder.create_model()

        # Step 14: Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate'], 
                           beta_1=self.config['beta_1'], 
                           beta_2=self.config['beta_2'], 
                           epsilon=self.config['epsilon']), 
            loss='binary_crossentropy', 
            metrics=[AUC(name='auc')]
        )

    def train_model(self, x_train, y_train):
        # Step 15: Train the model
        trainer = ModelTrainer(self.model, self.config)
        self.history = trainer.train(x_train, y_train)

    def save_results(self, save_architecture=False):
        # Step 16: Plot the results
        results_handler = ResultsHandler(self.model, self.history, self.path,self.config['model_name'])
        results_handler.plot_results()
        
        # Step 17: Save the results
        results_handler.save_results(save_architecture)
            
    def run(self, use_train_file=False, save_architecture=False):
        if use_train_file:
            # Directly read and prepare data from train file
            x_train, y_train = self.file_based_data_preparation()
        else:
            # Reprocess raw data
            merged_df = self.read_and_process_data()
            x_train, y_train = self.split_and_prepare_data(merged_df)
        
        # Execute the training pipeline
        self.build_model()
        self.train_model(x_train, y_train)
        self.save_results(save_architecture)

if __name__ == "__main__":
    config = {
        'file_path': r'D:\project',
        'checkpoint_path': r'D:\project',
        'user_feature_dim': 5,
        'song_feature_dim': 15,
        'contextual_feature_dim': 18,
        'user_embedding_dim': 32,
        'song_embedding_dim': 64,
        'epochs': 50,
        'batch_size': 256,
        'validation_split': 0.2,
        'patience': 5,
        'learning_rate': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-07,
        'model_name': 'test_model'
    }
    recommender_model = RecommendationModel(config)
    recommender_model.run(use_train_file=False, save_architecture=False)