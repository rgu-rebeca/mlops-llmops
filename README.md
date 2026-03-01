# MLOps Practice – Airbnb Price Classification

## Description

This project implements a simple Machine Learning workflow following MLOps practices, including data preprocessing, model training, experiment tracking with MLflow, and execution from the command line.

The dataset used is the **Airbnb listings dataset**.  
Originally, this dataset is used for a regression task to predict the price.  
In this practice, the target variable was modified to create a **classification problem** by grouping the price into three categories:

- low  
- mid  
- high  

This allows the model to perform multi-class classification instead of regression.


## Project Structure
├── main.py
├── data.py
├── preprocess.py
├── train.py
├── airbnb-listings-extract.csv
├── fastapi_app/
└── README.md


- **data.py** → load dataset and create target variable  
- **preprocess.py** → train/test split and preprocessing pipeline  
- **train.py** → model training and MLflow logging  
- **main.py** → command line entry point  


## Preprocessing

Steps applied:

- Remove invalid prices
- Create classification target using quantiles
- Train / test split
- Numerical pipeline  
  - SimpleImputer (median)  
  - StandardScaler  
- Categorical pipeline  
  - SimpleImputer (most frequent)  
  - OneHotEncoder  

Implemented using:
- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder


## Models

Supported models:

- LogisticRegression
- RandomForestClassifier

Run from command line:
python main.py <Model> <Hyperparameter>


Examples:
python main.py RandomForest 50
python main.py LogisticRegression 2000


Where the second argument represents:

- n_estimators for RandomForest
- max_iter for LogisticRegression


## MLflow

MLflow is used to track experiments.

Logged information:

- parameters
- metrics
- artifacts
- trained model

Metrics:

- accuracy
- precision_macro
- recall_macro
- f1_macro

Artifacts:

- confusion matrix
- saved model


## Notes

During the cloud build process, several permission errors appeared.  
Permissions had to be granted to the following services:

- Service Account  
- Cloud Build  
- Logging  
- Storage  

After fixing the permissions, the Docker image was successfully built and uploaded to the cloud.
