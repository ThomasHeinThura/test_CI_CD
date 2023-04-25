import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature
from train import eval_metrics

import sys
sys.path.append("src/data/")
sys.path.append("../data/")
from params_loader import read_params
from data_function import get_feat_and_target
from check_data_exist import check_dataset_exist




def select_run_id():
    df = mlflow.search_runs(filter_string="metrics.f1 > 0.945")

    return df 

def features_labels_split(test_data_path, target):
    test_dataset = pd.read_parquet(test_data_path)
    test_features,test_label = get_feat_and_target(test_dataset, target)
    return test_features, test_label

def predict_save_model(config_path, sampling):
    config = read_params(config_path)
    
    oversampling_test_data_path = config["processed_data_config"]["oversampling_test_data_csv"]
    undersampling_test_data_path = config["processed_data_config"]["undersampling_test_data_csv"]
    target = config["train_test_config"]["target"]
    
    check_dataset_exist(config_path)
    
    if sampling == 'undersampling': 
        test_data_path = undersampling_test_data_path
        
    if sampling == 'oversampling':
        test_data_path = oversampling_test_data_path
    
    test_features, test_label = features_labels_split(test_data_path, target)
    
    df = select_run_id()
    print("Loading the RUN ID")
    for i in range(len(df)):
        # search run id
        run_id = df['run_id'][i]
        print(f'Run_id is :{run_id}')
        model = df['params.Model'][i]
        
        logged_model = f'runs:/{run_id}/{model}'

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on a Pandas DataFrame.
        print("Testing the saving models")
        predict = loaded_model.predict(test_features)
        # print(f'{model} prediciton is \n {predict}  \n '
    
    print(f' The loading the model is success')
        

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="model_params.yaml")
    args.add_argument("--sampling",
                    default='undersampling',
                    help='select sampling dataset')
    
    parsed_args = args.parse_args()
    predict_save_model(config_path = parsed_args.config,
                       sampling = parsed_args.sampling)
        
