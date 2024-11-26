import json
import os
import shutil
from typing import Dict

import joblib
import kagglehub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from kagglehub.config import set_kaggle_credentials
from scipy.stats import ks_2samp
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

from constants import DATA_PATH, LABEL, PRODUCTION_EXPERIMENT_ID, PRODUCTION_RUN_ID


def download_data() -> None:
    # Fetch Credentials
    credentials = json.load(open("credentials.json"))  # Format {"username":"...", "api_key":"..."}
    set_kaggle_credentials(**credentials)
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(DATA_PATH, exist_ok=True)
    path = kagglehub.dataset_download("andrewmvd/heart-failure-clinical-data")
    print(path)
    shutil.move(path, DATA_PATH)


def get_cols_from_df(df: pd.DataFrame, features) -> pd.DataFrame:
    return df[features + [LABEL]].copy(deep=True)


def compute_ks(y_true, y_pred):
    prob_positive = y_pred[y_true == 1]  # Probabilities for the positive class
    prob_negative = y_pred[y_true == 0]  # Probabilities for the negative class

    # Compute KS statistic using ks_2samp
    ks_stat, p_value = ks_2samp(prob_positive, prob_negative)
    return ks_stat


def compute_metrics(y_true, y_pred):
    return {
        "auc_score": roc_auc_score(y_true, y_pred, multi_class="ovr"),
        "ks": compute_ks(y_true, y_pred),
    }


def plot_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        lw=lw,
        color="darkorange",
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    lw = 2
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def plot_importance(model):
    importances = model.feature_importances_
    features = model.feature_names_in_
    indices = np.argsort(importances)[::-1]
    ax = sns.barplot(x=importances[indices], y=[features[i] for i in indices])
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    return ax


def save_pipeline(pipeline):
    # Salvar o melhor modelo como um arquivo .joblib
    joblib_file = "pipeline_model.joblib"
    joblib.dump(pipeline, joblib_file)

    return joblib_file


def save_to_mlflow(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    run,
    experiment_metadata: Dict,
) -> None:
    # Save Model
    model = pipeline.named_steps["model"]
    save_pipeline(pipeline)
    mlflow.log_artifact("pipeline_model.joblib")

    # Save Evaluation Metrics
    train_predictions = pipeline.predict_proba(X_train)[:, 1]
    test_predictions = pipeline.predict_proba(X_test)[:, 1]
    train_metrics = compute_metrics(y_train, train_predictions)
    test_metrics = compute_metrics(y_test, test_predictions)
    metrics = {}
    for key, value in train_metrics.items():
        metrics[f"train_{key}"] = round(value, 4)
    for key, value in test_metrics.items():
        metrics[f"test_{key}"] = round(value, 4)
    mlflow.log_metrics(metrics)

    # Save Predictions
    predictions = {
        "train_predictions": str(train_predictions.tolist()),
        "test_predictions": str(test_predictions.tolist()),
    }
    mlflow.log_dict(predictions, "predictions.json")

    # Save Feature Importance
    try:
        fig, ax = plt.subplots()
        ax = plot_importance(model)
        mlflow.log_figure(fig, "feature_importance.png")
    except:
        print("Could not save feature importance plot")
        pass

    # Save Confusion Matrix
    # fig, ax = plt.subplots()
    # test_predictions_labels = [np.argmax(x) for x in test_predictions]
    # matrix = confusion_matrix(y_test, test_predictions_labels)
    # sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # mlflow.log_figure(fig, "confusion_matrix.png")

    # Save Train Predictions Histograms
    fig, ax = plt.subplots()
    ax.set_title("Train Predictions Histogram")
    hist_predictions = pd.DataFrame({"label": y_train, "prediction": train_predictions})
    sns.histplot(x="prediction", data=hist_predictions, bins=20, hue="label", kde=True)
    mlflow.log_figure(fig, "train_predictions_histogram.png")

    # Save Test Predictions Histograms
    fig, ax = plt.subplots()
    hist_predictions = pd.DataFrame({"label": y_test, "prediction": test_predictions})
    ax.set_title("Test Predictions Histogram")
    sns.histplot(x="prediction", data=hist_predictions, bins=20, hue="label", kde=True)
    mlflow.log_figure(fig, "test_predictions_histogram.png")

    # Save Parameters
    mlflow.log_params(model.get_params())

    # Save Model Metadata
    mlflow.log_dict(experiment_metadata, "metadata.json")

    # Save Test Roc Auc Curve
    fig, ax = plt.subplots()
    plot_roc_auc(y_test, test_predictions)
    mlflow.log_figure(fig, "roc_auc_curve_test.png")


def download_production_model():
    path = mlflow.artifacts.download_artifacts(
        f"mlflow-artifacts:/{PRODUCTION_EXPERIMENT_ID}/{PRODUCTION_RUN_ID}/artifacts/pipeline_model.joblib"
    )
    shutil.move(path, "pipeline_model.joblib")


def plot_confusion_matrix(predictions, labels, threshold):
    fig, ax = plt.subplots()
    test_predictions_labels = [int(x > threshold) for x in predictions]
    matrix = confusion_matrix(labels, test_predictions_labels)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")


def plot_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
