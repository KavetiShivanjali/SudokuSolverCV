import sys
from dataclasses import dataclass
import os

import numpy as np 
import pandas as pd
import tensorflow as tf
import cv2 
from tensorflow import keras

from src.exception import CustomException
from src.logger import logging
# from src.utils import save_object
# from src.components.data_ingestion import DataIngestionConfig


@dataclass
class DataTransformationConfig:
    model_file_path = os.path.join('artifacts/','model.hf5')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def image_preprocessor(self,image,label):
        try:
            def preprocess_image(image,height,width):
                image = image.numpy()
                image = cv2.resize(image,height,width)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.convertScaleAbs(image)
                image = cv2.adaptiveThresholding(image,255,0,0,129.0)
                image = image/255.0
                image = np.expand_dims(image,axis = 3)
                return image
            
            def preprocess(image,label):
                image = tf.py_function(preprocess_image,[image],tf.float32)
                image.set_shape([150,150,1])
                return image,label
            
            return preprocess(image,label)
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path,val_path,test_path):
        try:
            # read the files in train, test, validation path
            # train_dir = train_path
            # val_dir = DataIngestionConfig.validation_data_path
            # test_dir = DataIngestionConfig.test_data_path

            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                train_path,
                shuffle = True,
                seed = 111,
                image_size = (150,150),
                batch_size = None,
                label_mode = 'int'
            )
            logging.info("Loaded train data into train_ds")
            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                val_path,
                shuffle = True,
                seed = 111,
                image_size = (150,150),
                batch_size = None,
                label_mode = 'int'
            )
            logging.info("Loaded train data into val_ds")
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                test_path,
                shuffle = True,
                seed = 111,
                image_size = (150,150),
                batch_size = None,
                label_mode = 'int'
            )
            logging.info("Loaded train data into test_ds")
            # logging.info("Reading train and test data completed")
            train_ds = train_ds.map(self.image_preprocessor,num_parallel_calls = tf.data.AUTOTUNE)
            logging.info("Preprocessed train data into train_ds")
            val_ds = val_ds.map(self.image_preprocessor,num_parallel_calls = tf.data.AUTOTUNE)
            logging.info("Preprocessed train data into val_ds")
            test_ds = test_ds.map(self.image_preprocessor,num_parallel_calls = tf.data.AUTOTUNE)
            logging.info("Preprocessed train data into test_ds")
        except Exception as e:
            raise CustomException(e,sys)
    
