import pandas as pd
import numpy as np
import sklearn
import logging
import warnings
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import yaml
import argparse
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from params_loader import read_params


def load_data(data_path):
    """
    load csv dataset from given path
    input: csv path 
    output:pandas dataframe 
    note: Only 6 variables are used in this model building stage for the simplicity.
    """
    print("........ Loading Data..........")
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    return df

# ------------------------------------------------------------------------------- #
def check_duplication(raw_dataset):
    #check duplication of data
    number_of_duplications = raw_dataset.duplicated().sum()
    print(f"The duplication found on dataset is : {number_of_duplications}")
    return number_of_duplications

def remove_duplication(raw_dataset):
    number_of_duplications = check_duplication(raw_dataset)
    if number_of_duplications == 0:
        return raw_dataset
    else :
        raw_dataset.drop_duplicates(keep=False,inplace=True)
        remove_duplication(raw_dataset)
# ------------------------------------------------------------------------------- #
def check_skewness(raw_dataset):
    #Check skewness
    # this can check relation between each column
    skew_limit=0.75
    skew_value=raw_dataset[raw_dataset.columns].skew()
    # print(skew_value)
    skew_col=skew_value[abs(skew_value)>skew_limit]
    cols=skew_col.index
    return cols

def powertransform_skewness(raw_dataset):
    print("Remove duplication")
    remove_dupli_raw_dataset = remove_duplication(raw_dataset)
    print("Check skewness")
    skewed_col_nm = check_skewness(raw_dataset)
    # print(f"Skewness col are : \n {skewed_col_nm}")
    from sklearn.preprocessing import PowerTransformer
    pt=PowerTransformer(standardize=False)
    raw_dataset[skewed_col_nm]=pt.fit_transform(raw_dataset[skewed_col_nm])
# ------------------------------------------------------------------------------- #
def get_feat_and_target(raw_dataset,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    powertransform_dataset = powertransform_skewness(raw_dataset)
    X=raw_dataset.drop(target,axis=1)
    y=raw_dataset[target]
    print("spliiting X and y")
    return X,y   

def standardize_feature_labels(raw_dataset,target):
    X,y = get_feat_and_target(raw_dataset, target)
    #Normalize features
    sc=StandardScaler()
    scaled_X=sc.fit_transform(X)
    print("Standardizating")
    #Label Encode label
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)
    return scaled_X, y_transformed
# ------------------------------------------------------------------------------- #
def change_to_pandas(raw_dataset,scaled_X, y_transformed, target):
    # change feature to pandas
    print("Reframe to Pandas")
    data_columns = raw_dataset.drop(target,axis=1)
    scaled_X = pd.DataFrame(scaled_X, columns=data_columns.columns)
    # change labels to pandas
    y_transformed = pd.DataFrame(y_transformed)
    y_transformed.columns = ['Class']
    # zip two dataset.
    clean_dataset =pd.concat([scaled_X, y_transformed],ignore_index=False,axis=1,sort=False)
    return clean_dataset

def save_clean_data(raw_dataset,target, clean_data_path):
    scaled_X, y_transformed = standardize_feature_labels(raw_dataset, target)
    #clean raw dataset and save clean data
    clean_dataset = change_to_pandas(raw_dataset, scaled_X, y_transformed, target)
    print("Finish cleaning and save to path.")
    clean_dataset.to_csv(clean_data_path,index=False)

def clean_raw_data(config_path):
    """
        cleaning the dataset 
        selecting the features and labels
        Normalize the dataset
        clean duplication
    """
    # take parameters from config file
    config=read_params(config_path)
    raw_data_path=config["raw_data_config"]["raw_data_csv"]
    clean_data_path=config["clean_data_config"]["clean_data_csv"]
    target = config["train_test_config"]["target"]
    
    # load data
    creditcard_df=load_data(raw_data_path)
    print("START CLEANING")
    save_clean_data(creditcard_df, target, clean_data_path)
    print("Finish all processes")

    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    clean_raw_data(config_path=parsed_args.config)
