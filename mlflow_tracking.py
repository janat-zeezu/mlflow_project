# Simple MLflow tracking example
from mlflow import log_metric, log_param, log_artifact

if __name__ == "__main__":
    # Log parameters
    log_param("threshold", 2)
    log_param("verbosity", "DEBUG")

    # Log metrics
    log_metric("timestamp", 1000)
    log_metric("time_to_complete", 23)

    # Create and log an artifact
    with open("produced-dataset.csv", "w") as f:
        f.write("This is sample data")

    log_artifact("produced-dataset.csv")
