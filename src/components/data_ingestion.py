import os
import sys
import shutil
from src.exception import CustomException
from src.logger import logging

import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','/data/train/')
    test_data_path: str = os.path.join('artifacts','/data/test/')
    validation_data_path: str = os.path.join('artifacts','/data/validation/')
    raw_data_path: str = os.path.join('artifacts','/data/')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            for i in range(1,10):
                destination_train_folder = self.ingestion_config.train_data_path + str(i) + '/'
                destination_validation_folder = self.ingestion_config.validation_data_path + str(i) +'/'
                destination_test_folder = self.ingestion_config.test_data_path + str(i) + '/'
                os.makedirs(os.path.dirname(destination_train_folder),exist_ok = True)
                os.makedirs(os.path.dirname(destination_validation_folder),exist_ok = True)
                os.makedirs(os.path.dirname(destination_test_folder),exist_ok = True)
                logging.info("created train, validation and test folders")

                data_files = os.listdir('final_digits_data/'+str(i)+'/')
                train_cut = 70*len(data_files)//100
                val_cut = train_cut + 20*len(data_files)//100
                test_cut = val_cut + 10*len(data_files)//100
                train_files = data_files[:train_cut]
                val_files = data_files[train_cut:val_cut]
                test_files = data_files[val_cut:test_cut]

                for file in train_files:
                    shutil.move('final_digits_data/' + str(i)+'/'+file,destination_train_folder)
                
                for file in val_files:
                    shutil.move('final_digits_data/' + str(i) + '/' + file, destination_validation_folder)
                
                for file in test_files:
                    shutil.move('final_digits_data/' + str(i) + '/' + file, destination_test_folder)

            logging.info("Data ingested along with train test split.")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
