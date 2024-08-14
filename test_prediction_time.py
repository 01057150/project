from tensorflow.keras.models import load_model
from recommendation_model import PredictionProcessor
import time

user_id = 15796

model = load_model(r'D:\best_model_v4.h5')

model.summary()

batch_sizes = range(150000, 80000, -10000)

batch_size_times = []

for batch_size in batch_sizes:
    execution_times = []
    prep_times = []
    avg_prep_times = []
    avg_predict_times = []
    result_prep_times = []
    total_times = []
    
    print(f"\nTesting with batch size: {batch_size}")
    
    for i in range(10):
        start_time = time.time()
        
        result_df, data_prep_time, average_prep_time, average_predict_time, result_prep_time, total_time = \
            PredictionProcessor.predict_for_user_time(user_id, model, batch_size=batch_size)
        
        elapsed_time = time.time() - start_time
        
        execution_times.append(elapsed_time)
        prep_times.append(data_prep_time)
        avg_prep_times.append(average_prep_time)
        avg_predict_times.append(average_predict_time)
        result_prep_times.append(result_prep_time)
        total_times.append(total_time)
        
        print(f'User {user_id} recommendations (batch size {batch_size}):')
        print(result_df.head(10))
        print(f'Time taken for user {user_id} with batch size {batch_size}: {elapsed_time:.4f} seconds')
    
    average_execution_time = sum(execution_times) / len(execution_times)
    average_prep_time = sum(prep_times) / len(prep_times)
    average_batch_prep_time = sum(avg_prep_times) / len(avg_prep_times)
    average_batch_predict_time = sum(avg_predict_times) / len(avg_predict_times)
    average_result_prep_time = sum(result_prep_times) / len(result_prep_times)
    average_total_time = sum(total_times) / len(total_times)

    batch_size_times.append({
        "batch_size": batch_size,
        "average_execution_time": average_execution_time,
        "average_prep_time": average_prep_time,
        "average_batch_prep_time": average_batch_prep_time,
        "average_batch_predict_time": average_batch_predict_time,
        "average_result_prep_time": average_result_prep_time,
        "average_total_time": average_total_time
    })

print("\nBatch size vs. Average time metrics:")
for times in batch_size_times:
    print(f"\nBatch size {times['batch_size']}:")
    print(f"  Average total execution time: {times['average_execution_time']:.4f} seconds")
    print(f"  Average data preparation time: {times['average_prep_time']:.4f} seconds")
    print(f"  Average batch preparation time: {times['average_batch_prep_time']:.4f} seconds")
    print(f"  Average batch prediction time: {times['average_batch_predict_time']:.4f} seconds")
    print(f"  Average result preparation time: {times['average_result_prep_time']:.4f} seconds")
    print(f"  Average total predict_for_user time: {times['average_total_time']:.4f} seconds")
