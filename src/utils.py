import os
import sys
import pickle
import folium
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def map_model(df,kmeans,high_booking_areas):
    try:        
        map_obj = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)   

        for _, row in df.iterrows():
            lat, lng = row['latitude'], row['longitude']
            marker_color = 'blue' if kmeans.predict([[lat, lng]]) in high_booking_areas else 'green'
            marker = folium.CircleMarker(location=[lat, lng], color=marker_color, radius=5, fill=True, fill_color=marker_color)
            marker.add_to(map_obj)   

        return map_obj     

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)