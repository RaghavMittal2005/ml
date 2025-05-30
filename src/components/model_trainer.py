import os
import sys
from src.exception import customException
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainConfig:
    model_path=os.path.join('artifacts','model.pkl')

class ModelTrain:
    def __init__(self):
        self.config = ModelTrainConfig()
    def train(self,train_arr,test_arr):
      try:
        X_train, y_train, X_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
        model={
            'LinearRegression':LinearRegression(),
            'SVR':SVR(),
            'DecisionTreeRegressor':DecisionTreeRegressor(),
            'CatBoostRegressor':CatBoostRegressor(),
            'XGBRegressor':XGBRegressor(),
            'RandomTreeRegressor':RandomForestRegressor(),
            'AdaBoostRegressor':AdaBoostRegressor()
        }

        model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_dict=model)
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
        best_model = model[best_model_name]
        save_object(best_model,self.config.model_path)
        y_pred=best_model.predict(X_test)
        return r2_score(y_test,y_pred)
      except Exception as e:
        raise customException(e,sys)
    

            
        