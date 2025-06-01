import sys
from dataclasses import dataclass, field
from src.exception import customException
import os
from src.utils import load_obj
import pandas as pd

class PredictData:
    def __init__(self):
        pass
    
    def pred(self,feat):
        try:

            preprocess_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            model = load_obj(model_path)
            preprocessor = load_obj(preprocess_path)
            data = preprocessor.transform(feat)
            prediction = model.predict(data)
            return prediction
            
        except Exception as e:
            raise customException(e,sys)
class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
    def get_data(self):
        try:
            datainput={
                "gender":[self.gender],
                "race/ethnicity":[self.race_ethnicity],
                "parental level of education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test preparation course":[self.test_preparation_course],
                "reading score":[self.reading_score],
                "writing score":[self.writing_score]
            }
            return pd.DataFrame(datainput)
        except Exception as e:
            raise customException(e,sys)