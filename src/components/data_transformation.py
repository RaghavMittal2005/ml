import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import customException
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    pickle_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransform:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def data_pipeline(self):
        try:
            num_feat=['reading score', 'writing score']
            col_feat=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            col_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            pipe=ColumnTransformer(
                [
                    ('num',num_pipeline,num_feat),
                    ('col',col_pipeline,col_feat)
                ]
            )
            return pipe

        except Exception as e:
            raise customException(e,sys)
        
    def initiate_preprocessing(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            target='math score'
            pipe=self.data_pipeline()
            target_train=train_df[target]
            tr=train_df.drop(columns=[target],axis=1)
            test_train=test_df[target]
            te=test_df.drop(columns=[target],axis=1)

            logging.info("Read Data")
            train_arr=pipe.fit_transform(tr)
            test_arr=pipe.transform(te)

            intrain_arr=np.c_[train_arr,np.array(target_train)]
            
            intest_arr=np.c_[test_arr,np.array(test_train)]
            
            save_object(
                file_path=self.data_transformation_config.pickle_path,
                obj=pipe
            )
            logging.info("Fit and Transform Data")
            return intrain_arr,intest_arr



        except Exception as e:
            raise customException(e,sys)