from model import fit
from versioning import init_mlflow

if __name__ == "__main__":
    print("-- Starting pipeline")
    init_mlflow()
    fit()