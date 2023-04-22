# Undersampling the dataset
# import transform dataset 
# output undersampling dataset

from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from params_loader import read_params

# function to output undersampling csv 
def undersampling_dataset(clean_dataset, undersampling_data_path):
    """
    undersampling the dataset
    """
    scaled_X,transform_y = clean_dataset
    # Create an instance of NearMiss
    nm = NearMiss()
    # Fit and apply NearMiss to downsample the majority class
    nm_features, nm_labels = nm.fit_resample(scaled_X, y_transformed)
    undersampling_data = (nm_features, nm_labels)
    undersampling_data.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    
    
def undersampling_and_saved_data(config_path):
    """
    undersampling the dataset
    input: config path 
    output: save undersampling data files in output folder
    """
    config = read_params(config_path)
    
    #clean_dataset_path
    clean_data_path = config["clean_data_config"]["clean_data_csv"]
    #sampling path
    undersampling_data_path = config["sampling_data_config"]["undersampling_data_csv"] 
    
    # Read data from clean dataset
    clean_dataset =pd.read_csv(clean_data_path)
    
    undersampling_dataset(clean_dataset,
                         undersampling_data_path)
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    undersampling_and_saved_data(config_path=parsed_args.config)
    






    
    



    



