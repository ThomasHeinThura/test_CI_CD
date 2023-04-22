import pandas as pd
import numpy as np
# SKlearn model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import  RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from catboost import Pool, CatBoostClassifier, cv
import lightgbm as lgb
import xgboost as xgb
# Evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
# MLflow
import mlflow
import mlflow.sklearn
# others
import logging
import warnings
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#oversampling train/test path
oversampling_train_data_path = config["processed_data_config"]["oversampling_train_data_csv"]
oversampling_test_data_path = config["processed_data_config"]["oversampling_test_data_csv"]
#undersampling train/test path
undersampling_train_data_path = config["processed_data_config"]["undersampling_train_data_csv"]
undersampling_test_data_path = config["processed_data_config"]["undersampling_test_data_csv"]
#train/test config
split_ratio = config["train_test_config"]["train_test_split_ratio"]
random_state = config["train_test_config"]["random_state"]

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    with mlflow.start_run():
        # Model
        #Building Model Dict
        Models = {
            "Logistic Regression": LogisticRegression(),                    #
            "Support Vector Classifier": SVC(),                             # Ridge, SVC, LinearSVC, Passive_AC
            "Decision Tree": DecisionTreeClassifier(max_depth=6),           #
            "KNearest": KNeighborsClassifier(n_neighbors=5),                # doesn't have model.predict_proba so I left out.
            "GaussianNB" : GaussianNB(),                                    #
            "LDA" : LinearDiscriminantAnalysis(),                           # 
            "Ridge" : RidgeClassifier(),                                    #  
            "QDA" : QuadraticDiscriminantAnalysis(),                        #
            "Bagging" : BaggingClassifier(),                                #
            "MLP" : MLPClassifier(),                                        #
            "LSVC" : LinearSVC(),                                           #  
            "BernoulliNB" : BernoulliNB(),                                  #  
            "Passive_AC" : PassiveAggressiveClassifier(),                   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
            "SGB"     : GradientBoostingClassifier(n_estimators=100, random_state=9),
            "Adaboost" : AdaBoostClassifier(n_estimators=100, random_state=9, algorithm='SAMME.R', learning_rate=0.8),
            "Extra_T" : ExtraTreesClassifier(n_estimators=100, max_features=3),
            "R_forest" : RandomForestClassifier(max_samples=0.9, n_estimators=100, max_features=3),
            "XGB" : xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
            }
        
        for Model_Name, classifier in Models.items():
            i = 1
            print(f"{i}. {Model_Name}")
            # fit the model
            classifier.fit(train_nm_features, train_nm_labels)
            
            # make prediction
            predictions   = classifier.predict(test_nm_features)
            
            # calculate values
            base_score   = classifier.score(test_nm_features,test_nm_labels)
            accuracy     = accuracy_score(test_nm_labels, predictions)
            acc_bal      = balanced_accuracy_score(test_nm_labels, predictions)
            av_precision = average_precision_score(test_nm_labels, predictions)
            recall       = recall_score(test_nm_labels, predictions)#Set df_used to the fraudulent transactions' dataset.
            f1           = f1_score(test_nm_labels, predictions)
            mcc          = matthews_corrcoef(test_nm_labels, predictions)
            score = {
                "base_score"     : round(base_score,3),
                "accuary"        : round(accuracy,3),
                "acc_bal"        : round(acc_bal,3),
                "av_precision"   : round(av_precision,3),
                "recall"         : round(recall,3),
                "f1"             : round(f1,3),
                "mcc"            : round(mcc,3)   }
            
            with mlflow.start_run(nested=True):
                mlflow.log_param("Model"          , Model_Name)
                mlflow.log_param("base_score"     , base_score)
                mlflow.log_param("accuary"        , accuracy)
                mlflow.log_param("acc_bal"        , acc_bal)
                mlflow.log_param("av_precision"   , av_precision)
                mlflow.log_param("recall"         , recall)
                mlflow.log_param("f1"             , f1)
                mlflow.log_param("mcc"            , mcc)
            
            i = i+1
            print(f"{score}")
            print("________________________________________")
        
        
    

### Add lgb, catboost and Stacking and ANN


### virtualize end result

