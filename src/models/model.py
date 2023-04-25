import pandas as pd
import numpy as np
# SKlearn model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import  RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from catboost import Pool, CatBoostClassifier, cv
import lightgbm as lgb
import xgboost as xgb
# Evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,confusion_matrix,classification_report
# MLflow
import mlflow
import mlflow.sklearn
# others
import json
import yaml
import joblib
import argparse
from urllib.parse import urlparse
import logging
import warnings
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../data/')
from data_function import get_feat_and_target
from params_loader import read_params
warnings.filterwarnings("ignore")
np.random.seed(40)

def accuracymeasure(test_labels, predictions, avg_method):
    base_score   = classifier.score(test_nm_features,test_labels)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average=avg_method)
    recall = recall_score(test_labels, predictions, average=avg_method)
    f1score = f1_score(test_labels, predictions, average=avg_method)
    target_names = ['0','1']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(test_labels, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(test_labels, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Base score: ", base_score)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return base_score,accuracy,precision,recall,f1score

def features_labels_split(train_data_path, test_data_path, target):
    train_dataset = pd.read_csv(train_data_path, sep=",")
    test_dataset = pd.read_csv(test_data_path, sep=",")
    train_features,train_label = get_feat_and_target(train_dataset, target)
    test_features,test_label = get_feat_and_target(test_dataset, target)
    return train_features, train_label, test_features, test_label


def train_and_evaluate(config_path):
    config = read_params(config_path)
    
    #oversampling train/test path
    oversampling_train_data_path = config["processed_data_config"]["oversampling_train_data_csv"]
    oversampling_test_data_path = config["processed_data_config"]["oversampling_test_data_csv"]
    
    #undersampling train/test path
    undersampling_train_data_path = config["processed_data_config"]["undersampling_train_data_csv"]
    undersampling_test_data_path = config["processed_data_config"]["undersampling_test_data_csv"]
    
    #train/test config
    split_ratio = config["train_test_config"]["train_test_split_ratio"]
    random_state = config["train_test_config"]["random_state"]
    target = config["train_test_config"]["target"]
    
    
    train_features, train_label, test_features, test_label = features_labels_split(train_data_path, test_data_path, target)




if __name__ == "__main__":

    with mlflow.start_run():
        # Model
        #Building Model Dict
        Models = {
            "Logistic Regression": LogisticRegression(),                    #
            "Support Vector Classifier": SVC(),                             # Ridge, SVC, LinearSVC, Passive_AC
            "Decision Tree": DecisionTreeClassifier(max_depth=6),           #
            "KNearest": KNeighborsClassifier(n_neighbors=5),                # doesn't have model.predict_proba so I left out.
            "GaussianNB" : GaussianNB(),                                    #
            "LDA" : LinearDiscriminantAnalysis(),                           # 
            "Ridge" : RidgeClassifier(),                                    #  
            "QDA" : QuadraticDiscriminantAnalysis(),                        #
            "Bagging" : BaggingClassifier(),                                #
            "MLP" : MLPClassifier(),                                        #
            "LSVC" : LinearSVC(),                                           #  
            "BernoulliNB" : BernoulliNB(),                                  #  
            "Passive_AC" : PassiveAggressiveClassifier(),                   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
            "SGB"     : GradientBoostingClassifier(n_estimators=100, random_state=9),
            "Adaboost" : AdaBoostClassifier(n_estimators=100, random_state=9, algorithm='SAMME.R', learning_rate=0.8),
            "Extra_T" : ExtraTreesClassifier(n_estimators=100, max_features=3),
            "R_forest" : RandomForestClassifier(max_samples=0.9, n_estimators=100, max_features=3),
            "XGB" : xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
            }
        
        for Model_Name, classifier in Models.items():
            i = 1
            print(f"{i}. {Model_Name}")
            # fit the model
            classifier.fit(train_nm_features, train_nm_labels)
            
            # make prediction
            predictions   = classifier.predict(test_nm_features)
            

            
            
            
            # calculate values
            
            
            
            score = {
                "base_score"     : round(base_score,3),
                "accuary"        : round(accuracy,3),
                "acc_bal"        : round(acc_bal,3),
                "av_precision"   : round(av_precision,3),
                "recall"         : round(recall,3),
                "f1"             : round(f1,3),
                "mcc"            : round(mcc,3)   }
            
            with mlflow.start_run(nested=True):
                mlflow.log_param("Model"          , Model_Name)
                mlflow.log_param("base_score"     , base_score)
                mlflow.log_param("accuary"        , accuracy)
                mlflow.log_param("acc_bal"        , acc_bal)
                mlflow.log_param("av_precision"   , av_precision)
                mlflow.log_param("recall"         , recall)
                mlflow.log_param("f1"             , f1)
                mlflow.log_param("mcc"            , mcc)
            
            i = i+1
            print(f"{score}")
            print("________________________________________")
        
        
     



################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        model = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        accuracy,precision,recall,f1score = accuracymeasures(test_y,y_pred,'weighted')

        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)
       
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="model_params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)









### Add lgb, catboost and Stacking and ANN


### virtualize end result

