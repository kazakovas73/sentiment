import os
from pathlib import Path
import mlflow
import mlflow.sklearn


def init_mlflow():
    mlflow.set_tracking_uri('http://localhost:5001')
    mlflow.set_experiment("sentiment_analysis")