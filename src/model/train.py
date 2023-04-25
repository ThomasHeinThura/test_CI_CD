import pandas as pd
import numpy as np
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
sys.path.append("src/data/")
from data_function import get_feat_and_target
from params_loader import read_params
from models import build_and_load_models
from check_data_exist import check_dataset_exist
import warnings
warnings.filterwarnings("ignore")
np.random.seed(40)


def eval_metrics(classifier, test_features, test_labels, avg_method):
    
    # make prediction
    predictions   = classifier.predict(test_features)
    base_score   = classifier.score(test_features,test_labels)
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
    train_dataset = pd.read_parquet(train_data_path)
    test_dataset = pd.read_parquet(test_data_path)
    train_features,train_label = get_feat_and_target(train_dataset, target)
    test_features,test_label = get_feat_and_target(test_dataset, target)
    return train_features, train_label, test_features, test_label

def choose_dataset_to_train():
    choose_dataset = int(input("choose dataset to train \n"
                               "1. undersampling \n"
                               "2. oversampling \n"))
    return choose_dataset

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
    
    check_dataset_exist(config_path)
    with mlflow.start_run():
        choose_dataset = choose_dataset_to_train()
        
        if (choose_dataset == 1 ):
            train_data_path = undersampling_train_data_path
            test_data_path = undersampling_test_data_path
            mlflow.log_param("Sampling" , "Undersampling Dataset")
            
        elif (choose_dataset == 2):
            train_data_path = oversampling_train_data_path
            test_data_path = oversampling_test_data_path
            mlflow.log_param("Sampling" , "Oversampling Dataset")
            
        else :
            print("Please choose the dataset again")
            choose_dataset_to_train()
        
        train_features, train_label, test_features, test_label = features_labels_split(train_data_path, 
                                                                                    test_data_path, 
                                                                                    target)
        
        Models = build_and_load_models()
    
    ################### MLFLOW ###############################
    
        for Model_Name, classifier in Models.items():
            i = 1
            print(f"{i}. {Model_Name}")
            # fit the model
            classifier.fit(train_features, train_label)
            i = i+1
            # Calculate the metrics
            base_score,accuracy,precision,recall,f1score = eval_metrics(classifier,
                                                                        test_features,
                                                                        test_label, 
                                                                        'weighted')
            
            with mlflow.start_run(nested=True):
                mlflow.log_param("Model"           , Model_Name)
                mlflow.log_metric("base_score"     , base_score)
                mlflow.log_metric("accuary"        , accuracy)
                mlflow.log_metric("av_precision"   , precision)
                mlflow.log_metric("recall"         , recall)
                mlflow.log_metric("f1"             , f1score)
            print("________________________________________")


    # mlflow_config = config["mlflow_config"]
    # remote_server_uri = mlflow_config["remote_server_uri"]

    # mlflow.set_tracking_uri(remote_server_uri)
    # mlflow.set_experiment(mlflow_config["experiment_name"])

    # with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
       
    #     tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

    #     if tracking_url_type_store != "file":
    #         mlflow.sklearn.log_model(
    #             model, 
    #             "model", 
    #             registered_model_name=mlflow_config["registered_model_name"])
    #     else:
    #         mlflow.sklearn.load_model(model, "model")
 
# if __name__=="__main__":
#     args = argparse.ArgumentParser()
#     args.add_argument("--config", default="model_params.yaml")
#     parsed_args = args.parse_args()
#     train_and_evaluate(config_path=parsed_args.config)







