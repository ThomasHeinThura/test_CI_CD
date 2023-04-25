# Undersampling the dataset
# import transform dataset 
# output undersampling dataset

from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from params_loader import read_params
from data_function import get_feat_and_target, change_to_pandas



# function to output undersampling parquet
def undersampling_dataset(clean_dataset, undersampling_data_path, target):
    """
    undersampling the dataset
    """
    scaled_X,y_transformed = get_feat_and_target(clean_dataset, target)
    # Create an instance of NearMiss
    nm = NearMiss()
    # Fit and apply NearMiss to downsample the majority class
    print("Undersampling by NearMiss")
    nm_features, nm_labels = nm.fit_resample(scaled_X, y_transformed)
    undersampling_data = change_to_pandas(clean_dataset, nm_features, nm_labels, target)
    print("Saving as Parquet")
    undersampling_data.to_parquet(undersampling_data_path, index=False)
    
    
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
    target = config["train_test_config"]["target"]
    
    # Read data from clean dataset
    clean_dataset =pd.read_parquet(clean_data_path)
    
    undersampling_dataset(clean_dataset,
                         undersampling_data_path,
                         target)
    
    print("Finish")
    
# if __name__=="__main__":
#     args = argparse.ArgumentParser()
#     args.add_argument("--config", default="params.yaml")
#     parsed_args = args.parse_args()
#     undersampling_and_saved_data(config_path=parsed_args.config)
    






    
    



    



