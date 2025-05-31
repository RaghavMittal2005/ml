import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score as r2s
from src.exception import customException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

def save_object(obj,file_path):
 try:
    file=os.path.dirname(file_path)
    os.makedirs(file,exist_ok=True)
    with open(file_path, 'wb') as output:  # Overwrites any existing file
        dill.dump(obj, output)
 except  Exception as e:
    raise customException(e,sys)
 
def evaluate_model(X_train,y_train,X_test,y_test,model_dict,param):
   try:
      report={}
      for i in range(len(list(model_dict))):
        
        model = list(model_dict.values())[i]
        para=param[list(model_dict.keys())[i]]
        gs=GridSearchCV(model,param_grid=para,cv=5,verbose=1)
        gs.fit(X_train,y_train)
        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)
        y_pred_train=model.predict(X_train)
        y_pred_test=model.predict(X_test)
        r2_train=r2s(y_train,y_pred_train)
        r2_test=r2s(y_test,y_pred_test)
        report[list(model_dict.keys())[i]]=r2_test
        logging.info(f"Model {list(model_dict.keys())[i]} : {r2_test}")
      return report
   except Exception as e:
      raise customException(e,sys)
    