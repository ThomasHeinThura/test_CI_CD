# oversampling the dataset
# import transform dataset 
# output oversampling dataset

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from params_loader import read_params

# function to output oversampling csv 
def oversampling_dataset(clean_dataset, oversampling_data_path):
    scaled_X,transform_y = clean_dataset
    smote=SMOTE()
    sm_features,sm_labels=smote.fit_resample(scaled_X,transform_y)
    oversampling_data = (sm_features, sm_labels)
    oversampling_data.to_csv(oversampling_data_path, sep=",", index=False, encoding="utf-8")

def oversampling_and_saved_data(config_path):
    """
    oversampling the dataset
    input: config path 
    output: save oversampoing data files in output folder
    """
    config = read_params(config_path)
    
    #clean_dataset_path
    clean_data_path = config["clean_data_config"]["clean_data_csv"]
    #sampling path
    oversampling_data_path = config["sampling_data_config"]["overampling_data_csv"]
    
    # Read data from clean dataset
    clean_dataset =pd.read_csv(clean_data_path)
    
    oversampling_dataset(clean_dataset,
                         oversampling_data_path)
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    oversampling_and_saved_data(config_path=parsed_args.config)
    



    



