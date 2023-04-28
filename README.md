# The project is under development

The whole project draft line. 
The project contains 18 `scikit learn` models. you can check all models in [`src/model/models.py`](src/model/models.py).  And CI pipeline starts with integration with `git`.

1. Take input data and run the code with `python main --sampling undersampling` to use the undersampling dataset to train 18 models. If you want to train with the oversampling dataset just add oversampling `python main --sampling oversampling`.
2. Then, it automatically cleans the datasets and saves outputs as parquet files and stores them in the data folder. 
3. The clean data go through oversampling with `SMOTE` and undersampling with `NearMiss` and then, prepare for training and testing the data.
4. After preprocessing the data, then model training start.
5. The training model results are viewed with `MLflow`. To check the result just simply type `mlflow ui`.
6. The training process is checked with the `prefect`. 
7. If the test matrices score f1 is more than 0.945, then the models are automatically saved with run id in `mlartifacts` folder. 
8. Then the save models are loaded with run_id for further staging like registering and testing stage or production deployment stage.


The CD pipeline start with configuration with GitHub action and 
1. retesting the model with more data. 
2. monitoring the model with `EvidentlyAI` and `Grafana` with `docker`for model degeneration, data degeneration, data drifting and furthermore. 
3. Then the whole process is updated to the server or docker for further use age. 

Further more adding. 
* want to store data train and testing plots to mlflow database.
* want to upload to the website and build the website. 
* Write unit testing and automatic check for code quality.