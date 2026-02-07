# Script for training the XGBoost model

# For data manipulation
import pandas as pd
import sklearn

# For creating folders
import os

# For data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# For model training, tuning, and evaluation
# XGBoost classifier
import xgboost as xgb 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

# For model serialization
import joblib

# For accessing HuggingFace Hub
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Params to track the experiment in MLFlow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Defining HF API
api = HfApi()

# Defining dataset paths in HF
Xtrain_path = "hf://datasets/KavinPrasathK/Tourism_Package_Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/KavinPrasathK/Tourism_Package_Prediction/Xtest.csv"
ytrain_path = "hf://datasets/KavinPrasathK/Tourism_Package_Prediction/ytrain.csv"
ytest_path = "hf://datasets/KavinPrasathK/Tourism_Package_Prediction/ytest.csv"

# Loading the data from HF
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# One-hot encode categorical and scale numeric features
numeric_features = [
    'Age',  
    'CityTier', 
    'DurationOfPitch', 
    'NumberOfPersonVisiting', 
    'NumberOfFollowups', 
    'PreferredPropertyStar',
    'NumberOfTrips', 
    'Passport', 
    'PitchSatisfactionScore', 
    'OwnCar', 
    'NumberOfChildrenVisiting', 
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'Gender',
    'Occupation',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]


# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Data preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost classifier model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid for tuning the model
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():

    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Logging all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Logging each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Logging best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Logging the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Serialize and save the best model 
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Logging the model artifact path
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Uploading to Hugging Face
    repo_id = "KavinPrasathK/Tourism_Package_Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")

    # Step 2: If repo doesn't exist : create repo     
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")
       
    api.upload_file(
        path_or_fileobj="best_tourism_package_prediction_model_v1.joblib",
        path_in_repo="best_tourism_package_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
