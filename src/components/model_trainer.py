import os 
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging
from src.utills import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path =os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split training and test input data')
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models  = {
                'Random forest': RandomForestRegressor(),
                'Decission tree':DecisionTreeRegressor(),
                'Gradient boosting':GradientBoostingRegressor(),
                'Linear regressor':LinearRegression(),
                'K-nieghbors classifier':KNeighborsRegressor(),
                'XGBclassifier':XGBRegressor(),
                'Catboosting classifier':CatBoostRegressor(),
                'AdaBoost classifier':AdaBoostRegressor()
            }
            params = {
                'Decission tree':{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                },
                'Random forest':{
                    'criterion':['squared_error','friedman_mse','poisson'],
                    'n_estimators':[5,16,32,64,128,256]
                },
                'Gradient boosting':{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[7,16,32,64,128,256]
                },
                'Linear regressor':{},
                'K-nieghbors classifier':{
                    'n_neighbors':[5,7,9,11]
                },
                'XGBclassifier':{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Catboosting classifier':{
                    'depth':[6,8,9,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                'AdaBoost classifier':{
                    'learning_rate':[0.1,.01,.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                        
                }     
                
                
                
            }
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
            
            ## to get the best model score 
            best_model_score = max(sorted(model_report.values()))
            ## to get name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise customException('No best model found')
            logging.info(f'best found model pn both training and testing dataset')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
                
            )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
            
        except Exception as e:
            raise customException(e,sys)
            
