import pandas as pd

def get_feat_and_target(clean_dataset,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    X=clean_dataset.drop(target,axis=1)
    y=clean_dataset[target]
    print("spliiting X and y")
    return X,y   

def change_to_pandas(clean_dataset,nm_features, nm_labels, target):
    # change feature to pandas
    print("Reframe to Pandas")
    data_columns = clean_dataset.drop(target,axis=1)
    nm_features = pd.DataFrame(nm_features, columns=data_columns.columns)
    # change labels to pandas
    nm_labels = pd.DataFrame(nm_labels)
    nm_labels.columns = ['Class']
    # zip two dataset.
    clean_dataset =pd.concat([nm_features, nm_labels],ignore_index=False,axis=1,sort=False)
    return clean_dataset