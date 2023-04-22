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


def load_data(data_path):
    """
    load csv dataset from given path
    input: csv path 
    output:pandas dataframe 
    note: Only 6 variables are used in this model building stage for the simplicity.
    """
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    return df

#Check skewness


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
    
def check_skewness(raw_dataset):
    # this can check relation between each column
    skew_limit=0.75
    skew_value=raw_dataset[raw_dataset.columns].skew()
    # print(skew_value)
    skew_col=skew_value[abs(skew_value)>skew_limit]
    cols=skew_col.index
    return cols

def powertransform_skewness(raw_dataset):
    skewed_col_nm = check_skewness(raw_dataset)
    print(f"Skewness col are : \n {skewed_col_nm}")
    from sklearn.preprocessing import PowerTransformer
    pt=PowerTransformer(standardize=False)
    raw_dataset[skewed_col_nm]=pt.fit_transform(raw_dataset[skewed_col_nm])

def standardize_dataset(raw):
    return 

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y   

def clean_raw_data(config_path):
    """
        These are 
    """
    # take parameters from config file
    config=read_params(config_path)
    raw_data_path=config["raw_data_config"]["raw_data_csv"]
    clean_data_path=config["clean_data_config"]["clean_data_csv"]
    
    # load data
    creditcard_df=load_data(raw_data_path)
    creditcard_df=remove_duplication(creditcard_df)
    powertransform_skewness(creditcard_df)
    
    
    
    
    
    #clean raw dataset and save clean data
    clean_data.to_csv(raw_data_path,index=False)
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)








# Feature and label
X = creditcard_df.drop(['Class'], axis=1)
y = creditcard_df['Class']
y.value_counts().to_frame()


from sklearn.preprocessing import StandardScaler
### Note that you can fit_transform the whole oversampled features(train+test) from begining
sc=StandardScaler()
scaled_X=sc.fit_transform(X)
scaled_X

from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

print(scaled_X.shape, y.shape)

### output transform dataset

