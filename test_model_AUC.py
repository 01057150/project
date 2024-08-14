import time
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score
from data_management import FileManage
from feature_processing import FeatureProcessor

# Step 1: Load the model
start_time = time.time()
model_path = r'D:\best_model_v6.h5'
model = load_model(model_path)
model.summary()
elapsed_time = time.time() - start_time
print(f"    Step 1:Time taken to load model: {elapsed_time:.4f} seconds")

# Step 2: Read and process the test data
start_time = time.time()
test_df = FileManage.read_files(file='test_file')
x_test, y_test = FeatureProcessor.one_hot_encoder(test_df, mode='test')
elapsed_time = time.time() - start_time
print(f"    Step 2:Time taken to read and process data: {elapsed_time:.4f} seconds")

# Step 3: Make predictions
start_time = time.time()
predictions = model.predict(x_test, batch_size=1000000)
elapsed_time = time.time() - start_time
print(f"    Step 3:Time taken to make predictions: {elapsed_time:.4f} seconds")

# Step 4: Create a DataFrame to view the prediction results
start_time = time.time()
results_df = pd.DataFrame({
    'msno_encoded': x_test[0],           # User ID
    'song_id_encoded': x_test[1],        # Song ID
    'prediction_target': predictions.flatten()  # Flatten to 1D array
})
elapsed_time = time.time() - start_time
print(f"    Step 4:Time taken to create DataFrame: {elapsed_time:.4f} seconds")

# Step 5: View the first few rows of the prediction results
start_time = time.time()
print(results_df.head())
elapsed_time = time.time() - start_time
print(f"    Step 5:Time taken to view DataFrame head: {elapsed_time:.4f} seconds")

# Step 6: Calculate AUC
start_time = time.time()
print('Prediction complete, calculating AUC')
auc = roc_auc_score(y_test, predictions)
elapsed_time = time.time() - start_time
print(f"AUC: {auc}")
print(f"    Step 6:Time taken to calculate AUC: {elapsed_time:.4f} seconds")
