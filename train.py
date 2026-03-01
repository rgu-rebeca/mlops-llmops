from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def train_model(
    model_name,
    hyperparam,
    preprocess,
    X_train,
    X_test,
    y_train,
    y_test,
):

    hyperparam = int(hyperparam)

    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=hyperparam)

    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=hyperparam,
            n_jobs=-1,
        )

    else:
        raise ValueError("Modelo no soportado")

    clf = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    mlflow.set_experiment("airbnb_price_classification")

    with mlflow.start_run():

        mlflow.log_param("model", model_name)
        mlflow.log_param("hyperparam", hyperparam)

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="macro")
        rec = recall_score(y_test, pred, average="macro")
        f1 = f1_score(y_test, pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        fig, ax = plt.subplots()

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            pred,
            ax=ax,
        )

        fig_path = "confusion_matrix.png"
        plt.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path)

        mlflow.sklearn.log_model(
            clf,
            "model",
        )

        print(acc, f1)