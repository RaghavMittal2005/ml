import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import customException
from dataclasses import dataclass
from src.components.data_transformation import DataTransform

@dataclass
class dataIngestionConfig:
    train_datapath:str =os.path.join('artifacts','train.csv')
    test_datapath:str =os.path.join('artifacts','test.csv')
    raw_datapath:str=os.path.join('artifacts','raw.csv')

class dataIngestion:
   def __init__(self):
       self.dataIngest=dataIngestionConfig()

   def read_data(self):
    try:
      df=pd.read_csv('notebook\data\StudentsPerformance.csv')   
      logging.info("Read Data")
      train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)
      os.makedirs(os.path.dirname(self.dataIngest.test_datapath),exist_ok=True)
      df.to_csv(self.dataIngest.raw_datapath,index=False,header=True)
      train_data.to_csv(self.dataIngest.train_datapath,index=False,header=True)
      test_data.to_csv(self.dataIngest.test_datapath,index=False,header=True)
      logging.info("Data Ingested")
      return train_data,test_data

    except Exception as e:
      raise customException(e,sys)
if __name__ == "__main__":
   obj=dataIngestion()
   train_data,test_data=obj.read_data()
   dataTrans=DataTransform()
   train_arr,test_arr=dataTrans.initiate_preprocessing(dataIngestionConfig.train_datapath,dataIngestionConfig.test_datapath)

   