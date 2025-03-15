import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Define model parameters
n_estimators = 100
max_depth = 7

import mlflow

# Set the correct tracking URI
mlflow.set_tracking_uri("file:///C:/Users/HP/mlFlow/mlflow_project/mlruns")  # Use forward slashes
mlflow.set_experiment("Iris Classification")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Set tracking URI and experiment
mlflow.set_tracking_uri("file:///C:/Users/HP/mlFlow/mlflow_project/mlruns")
mlflow.set_experiment("Iris Classification")


with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)  # ✅ Model is trained here

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Create input example (one sample from training set)
    example_input = pd.DataFrame([X_train[0]], columns=iris.feature_names)

    # ✅ Ensure model is trained before logging
    mlflow.sklearn.log_model(model, "model", input_example=example_input)

print("Training complete! Check MLflow UI for results.")


# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    mlflow.sklearn.log_model(model, "model")
