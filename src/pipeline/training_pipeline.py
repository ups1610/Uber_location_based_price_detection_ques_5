import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation() 
    ip_trained_df,target_feature_train_df,ip_test_df,target_feature_test_df,booking_tain_data,booking_test_data,_,_ = data_transformation.initaite_data_transformation(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(ip_trained_df,ip_test_df,target_feature_train_df,target_feature_test_df,booking_tain_data)
   
