# Basic Import
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import map_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path_cluster = os.path.join('artifacts','cluster.pkl')
    trained_model_file_path_model = os.path.join('artifacts','model.pkl')
    mapping_data = os.path.join('artifacts','map_data.html')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,input_feature_train_df,input_feature_test_df,target_feature_train_df,target_feature_test_df,booking_data_train):
        try:
            logging.info("Making cluster of latitude and longitude dataframe")
            kmeans = KMeans(n_clusters=3)  # Specify the number of high booking areas you want to identify
            kmeans.fit(booking_data_train)
            
            high_booking_areas = kmeans.predict(booking_data_train)
            print(high_booking_areas)
            logging.info(f"{high_booking_areas}")

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                input_feature_train_df,
                target_feature_train_df,
                input_feature_test_df,
                target_feature_test_df
            )
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            # training with best model as analysed in EDA with hyper-tuning parameters

            best_model=RandomForestRegressor()
            best_model.fit(X_train,y_train)


            rf_predictions = best_model.predict(X_test)

            print("===================== Model trained for highly booking areas price prediction ===================================")
            print("Random Forest Regressor:")
            # Calculate accuracy and confusion matrix
            r2_square = r2_score(y_test, rf_predictions)
            print(f"accuracy score : {r2_square}") 
            logging.info(f"accuracy score : {r2_square}")
          
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path_cluster,
                 obj=kmeans
            )  

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path_model,
                 obj=best_model
            )    

            logging.info("Mapping the data")
            map_obj = map_model(input_feature_train_df,kmeans,high_booking_areas)
            map_obj.save('artifacts/map.html')

            logging.info("Model trained successfully and saved and mapped")      
          
        except Exception as e:
            logging.info('Exception occured at Model Training of Revenue')
            raise CustomException(e,sys)


    