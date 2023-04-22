import pandas as pd

def get_feat_and_target(dataset,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    X=dataset.drop(target,axis=1)
    y=dataset[target]
    print("spliiting X and y")
    return X,y   

def change_to_pandas(dataset,features, labels, target):
    # change feature to pandas
    print("Reframe to Pandas")
    data_columns = dataset.drop(target,axis=1)
    features = pd.DataFrame(features, columns=data_columns.columns)
    # change labels to pandas
    labels = pd.DataFrame(labels)
    labels.columns = ['Class']
    # zip two dataset.
    dataset =pd.concat([features, labels],ignore_index=False,axis=1,sort=False)
    return dataset