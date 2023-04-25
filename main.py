import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append("src/model/")
from train import train_and_evaluate
from select_run_id import predict_save_model
sys.path.append("src/data/")
# from data_function import get_feat_and_target
from params_loader import read_params
# from models import build_and_load_models
# from check_data_exist import check_dataset_exist
import warnings
import argparse
import numpy as np
import mlflow

warnings.filterwarnings("ignore")
np.random.seed(40)

def main(config_path,sampling):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    # mlflow.set_experiment("my-experiment-1")
    config = read_params(config_path)
    train_and_evaluate(config_path, sampling)
    predict_save_model(config_path, sampling)
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="model_params.yaml")
    
    args.add_argument("--sampling",
                      default='undersampling',
                      help='select sampling dataset')
    
    parsed_args = args.parse_args()
    
    main(config_path = parsed_args.config,
         sampling = parsed_args.sampling)
    