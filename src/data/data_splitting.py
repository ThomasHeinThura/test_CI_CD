import argparse
import pandas as pd
from params_loader import read_params
from sklearn.model_selection import train_test_split

def split_data(df,train_data_path,test_data_path,split_ratio,random_state):
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_parquet(train_data_path, index=False)
    test.to_parquet(test_data_path, index=False)    

def split_sampling_and_saved_data(config_path):
    """
    split the train dataset(data/raw) and save it in the data/processed folder
    input: config path 
    output: save splitted files in output folder
    """
    config = read_params(config_path)
    #sampling path
    oversampling_data_path = config["sampling_data_config"]["oversampling_data_csv"]
    undersampling_data_path = config["sampling_data_config"]["undersampling_data_csv"] 
    #oversampling train/test path
    oversampling_train_data_path = config["processed_data_config"]["oversampling_train_data_csv"]
    oversampling_test_data_path = config["processed_data_config"]["oversampling_test_data_csv"]
    #undersampling train/test path
    undersampling_train_data_path = config["processed_data_config"]["undersampling_train_data_csv"]
    undersampling_test_data_path = config["processed_data_config"]["undersampling_test_data_csv"]
    #train/test config
    split_ratio = config["train_test_config"]["train_test_split_ratio"]
    random_state = config["train_test_config"]["random_state"]
    
    # Read data from sampling
    print("Loading dataset")
    oversampling_dataset=pd.read_parquet(oversampling_data_path)
    undersampling_dataset = pd.read_parquet(undersampling_data_path)
    
    print("Start splitting oversampling dataset")
    split_data(oversampling_dataset,
               oversampling_train_data_path,
               oversampling_test_data_path,
               split_ratio,
               random_state)
    
    print("Start splitting undersampling dataset")
    split_data(undersampling_dataset,
               undersampling_train_data_path,
               undersampling_test_data_path,
               split_ratio,
               random_state)
    
    print("Finish all processes")
    
# if __name__=="__main__":
#     args = argparse.ArgumentParser()
#     args.add_argument("--config", default="params.yaml")
#     parsed_args = args.parse_args()
#     split_sampling_and_saved_data(config_path=parsed_args.config)
    



