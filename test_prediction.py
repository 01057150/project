from tensorflow.keras.models import load_model
from recommendation_model import PredictionProcessor

user_ids = [7778, 15796, 11658] #新註冊使用者, 一般系統使用者, 資深系統使用者

model = load_model(r'D:\best_model_v4.h5')

model.summary()

for user_id in user_ids:
    
    result_df, data_prep_time, average_prep_time, average_predict_time, result_prep_time, total_time = \
        PredictionProcessor.predict_for_user_time(user_id, model,batch_size=120000)
    
    print(f'User {user_id} recommendations:')
    
    print(result_df.head(10))