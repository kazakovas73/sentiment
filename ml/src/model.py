from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data import split_datasets
from pathlib import Path
import pickle
import mlflow
from mlflow.models.signature import infer_signature


def save_model(model, path: Path):
    with open(path / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("-- Model is saved")

    
def load_model(path: Path):
    with open(path / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def fit():
    X_train, X_test, y_train, y_test = split_datasets(Path(__file__).parent.parent.parent / "data")

    print("-- Model fitting...")
    clf = RandomForestClassifier(criterion='gini', max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    print("\tSuccess")

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    metrics = {
        'train/accuracy': accuracy_score(y_train, y_pred_train),
        'train/f1': f1_score(y_train, y_pred_train, average='macro'),
        'train/precision': precision_score(y_train, y_pred_train, average='macro'),
        'train/recall': recall_score(y_train, y_pred_train, average='macro'),

        'test/accuracy': accuracy_score(y_test, y_pred_test),
        'test/f1': f1_score(y_test, y_pred_test, average='macro'),
        'test/precision': precision_score(y_test, y_pred_test, average='macro'),
        'test/recall': recall_score(y_test, y_pred_test, average='macro'),
    }

    signature = infer_signature(X_test, clf.predict(X_test))

    with mlflow.start_run():
        # Логируем параметры модели (например, число деревьев)
        mlflow.log_param("criterion", 'gini')
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)
        
        # Логируем метрики
        mlflow.log_metrics(metrics)
        
        # Логируем модель sklearn
        mlflow.sklearn.log_model(
            clf, 
            "random_forest_model",
            input_example=X_test[:3],
            signature=signature
        )

        run_id = mlflow.active_run().info.run_id

        # Регистрируем модель в Model Registry
        model_uri = f"runs:/{run_id}/random_forest_model"
        mlflow.register_model(model_uri, "RandomForestClassifierModel")

    print(metrics)

