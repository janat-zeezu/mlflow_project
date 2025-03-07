import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/IrisClassification")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow run
with mlflow.start_run():
    # Set parameters
    n_estimators = 100
    max_depth = 7
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("dataset", "iris")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, 
                               average='weighted')
    recall = recall_score(y_test, y_pred, 
                         average='weighted')
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log the model
    mlflow.sklearn.log_model(model, 
                            "random_forest_model")