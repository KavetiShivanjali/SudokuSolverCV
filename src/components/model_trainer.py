import os
import sys

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2

@dataclass
class ModelTrainerConfig:
    model_trainer_config = os.path.join('artifacts/','sratch_model.h5')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig
    
    def train_model(self, train_ds, val_ds, test_ds):
        try:
            X_train,y_train = ([],[])
            for image,label in train_ds:
                X_train.append(image)
                y_train.append(label)
            logging.info("Created X_train and y_train")

            X_val, y_val = ([],[])
            for image,label in val_ds:
                X_val.append(image)
                y_val.append(label)
            logging.info("Created X_val and y_val")

            X_test, y_test = ([],[])
            for image,label in test_ds:
                X_test.append(image)
                y_test.append(label)
            logging.info("Created X_test and y_test")

            Y_train = tf.keras.utils.to_categorical(y_train)
            Y_val = tf.keras.utils.to_categorical(y_val)
            Y_test = tf.keras.utils.to_categorical(y_test)

            Y_train = np.array([np.concatenate(([0],arr)) for arr in Y_train])
            Y_val = np.array([np.concatenate(([0],arr)) for arr in Y_val])
            Y_test = np.array([np.concatenate(([0],arr)) for arr in Y_test])

            # model is remaining
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape = (150,150,1)),
                tf.keras.layers.Conv2D(16,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.Conv2D(32,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.Conv2D(64,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.Conv2D(128,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(256,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(512,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(512,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(1024,3,padding = 'same',activation = 'relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512,activation = 'relu'),
                tf.keras.layers.Dense(256,activation = 'relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128,activation = 'relu'),
                tf.keras.layers.Dense(64,activation = 'relu'),
                tf.keras.layers.Dense(10,activation = 'softmax')
            ])
            logging.info("Configured CNN architecture")
            # model compilation
            model.compile(optimizer = 'adam',
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy'])
            logging.info("Compiled the CNN model")
            # model fit
            X_train = np.expand_dims(X_train, axis = 3)
            X_val = np.expand_dims(X_val, axis = 3)
            X_test = np.expand_dims(X_test, axis = 3)
            ckpt_path = os.path.join('artifacts/','checkpoint/')
            callbacks_1 = tf.keras.callbacks.ModelCheckpoint(ckpt_path,save_weights_only = True,monitor = 'val_loss',mode = 'min',save_best_only = True)
            callbacks_2 = tf.keras.callbacks.ModelCheckpoint(ckpt_path,save_weights_only = True, monitor = 'val_accuracy',mode = 'max',save_best_only = True)
            model_history = model.fit(X_train,Y_train,batch_size = 32,epochs = 15,
                                      validation_data =(X_val,Y_val),callbacks = [callbacks_1,callbacks_2])
            logging.info('Model has been trained')
            # save model
            model.save(self.model_trainer_config.model_trainer_config)
            logging.info('Model has been saved!')

        except Exception as e:
            raise CustomException(e,sys)

