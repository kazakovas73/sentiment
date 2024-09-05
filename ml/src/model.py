from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data import split_datasets
from pathlib import Path


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

    print(metrics)

