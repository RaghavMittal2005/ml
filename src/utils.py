import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import customException

def save_object(obj,file_path):
 try:
    file=os.path.dirname(file_path)
    os.makedirs(file,exist_ok=True)
    with open(file_path, 'wb') as output:  # Overwrites any existing file
        dill.dump(obj, output)
 except  Exception as e:
    raise customException(e,sys)
    