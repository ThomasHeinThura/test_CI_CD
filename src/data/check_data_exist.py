from data_cleaner import clean_raw_data
from oversampling import oversampling_and_saved_data
from undersampling import undersampling_and_saved_data
from data_splitting import split_sampling_and_saved_data
from params_loader import read_params
import argparse
from os.path import exists

def check_dataset_exist(config_path):
    config=read_params(config_path)
    # data path
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    clean_data_path = config["clean_data_config"]["clean_data_csv"]
    #sampling path
    oversampling_data_path = config["sampling_data_config"]["oversampling_data_csv"]
    undersampling_data_path = config["sampling_data_config"]["undersampling_data_csv"] 
    #oversampling train/test path
    oversampling_train_data_path = config["processed_data_config"]["oversampling_train_data_csv"]
    oversampling_test_data_path = config["processed_data_config"]["oversampling_test_data_csv"]
    #undersampling train/test path
    undersampling_train_data_path = config["processed_data_config"]["undersampling_train_data_csv"]
    undersampling_test_data_path = config["processed_data_config"]["undersampling_test_data_csv"]
    
    
    if exists(raw_data_path) == False:
        print("You don't have raw dataset. \n"
              "Download the dataset.")
    else :
        print("Raw data is present and continue checking")
        
        if exists(clean_data_path) == False:
            clean_raw_data(config_path)
        
        if exists(undersampling_data_path) == False:
            undersampling_and_saved_data(config_path)
        
        if exists(oversampling_data_path) == False:
            oversampling_and_saved_data(config_path)
        
        if exists(undersampling_train_data_path) == False:
            split_sampling_and_saved_data(config_path)
        
        if exists(undersampling_test_data_path) == False:
            split_sampling_and_saved_data(config_path)
        
        if exists(oversampling_train_data_path) == False:
            split_sampling_and_saved_data(config_path)
        
        if exists(oversampling_test_data_path) == False:
            split_sampling_and_saved_data(config_path) 
    
        else :
            print("All dataset are present.")
        
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    check_dataset_exist(config_path=parsed_args.config)

    
    