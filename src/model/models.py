
# SKlearn model
import sklearn
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
# from catboost import Pool, CatBoostClassifier, cv
# import lightgbm as lgb
import xgboost as xgb


def build_and_load_models():
    # Models
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
        # "XGB" : xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
        }
    return Models
 

### Add lgb, catboost and Stacking and ANN


### virtualize end result

