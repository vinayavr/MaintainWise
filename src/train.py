from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib

from src.data_pipeline import (
    ROOT, PROCESSED, ARTIFACTS, TARGET, FAILURE_MODE_COLS, ID_COLS,
    load_dataset, clean_and_engineer, model_features
)

EXPERIMENT_NAME = "MaintainWise-AI4I-Predictive-Maintenance"
MODEL_NAME = "MaintainWiseBestModel"
RANDOM_STATE = 42

def make_preprocessor(numeric, categorical, scale=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = SkPipeline(num_steps)
    categorical_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ], remainder="drop")

def plot_eda(df):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    df[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Machine failure class distribution")
    ax.set_xlabel("Machine failure (0 = No, 1 = Yes)")
    ax.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "class_distribution.png", dpi=150)
    plt.close(fig)

    numeric = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    for col in ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
            ax.set_title(f"{col} by machine-failure class")
            fig.tight_layout()
            fig.savefig(ARTIFACTS / f"boxplot_{col.split(' [')[0].replace(' ', '_').lower()}.png", dpi=150)
            plt.close(fig)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Re-read raw data, then run a deterministic pipeline.
    raw = load_dataset()
    df = clean_and_engineer(raw)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df.to_csv(PROCESSED / "ai4i_cleaned.csv", index=False)
    df.to_parquet(PROCESSED / "ai4i_cleaned.parquet", index=False)
    plot_eda(df)

    features, numeric, categorical = model_features(df)
    X = df[features].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # SMOTE runs only inside the training pipeline, never on the held-out test set.
    # Use a safe k_neighbors for the minority class available in the training fold.
    minority_count = int(y_train.value_counts().min())
    smote_k = max(1, min(5, minority_count - 1))
    preprocessor = make_preprocessor(numeric, categorical, scale=True)

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 5.0], "model__solver": ["liblinear"]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 3, 5]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 8, 14], "model__min_samples_leaf": [1, 3]}
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
            {"model__C": [0.5, 1.0, 3.0], "model__kernel": ["rbf", "linear"]}
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss",
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, n_jobs=2
            ),
            {"model__n_estimators": [100, 150], "model__max_depth": [3, 4], "model__learning_rate": [0.05, 0.1]}
        )
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []
    fitted_models = []
    for name, (estimator, param_grid) in models.items():
        print(f"\n=== Training {name} ===")
        pipe = ImbPipeline(steps=[
            ("preprocessor", make_preprocessor(numeric, categorical, scale=True)),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)),
            ("model", estimator)
        ])
        # A small randomized search to keep the deadline achievable.
        search = RandomizedSearchCV(
            pipe, param_distributions=param_grid, n_iter=min(3, np.prod([len(v) for v in param_grid.values()])),
            scoring="f1", cv=3, random_state=RANDOM_STATE, n_jobs=1,
            refit=True, error_score="raise"
        )
        with mlflow.start_run(run_name=name) as run:
            search.fit(X_train, y_train)
            best = search.best_estimator_
            pred = best.predict(X_test)
            proba = best.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba))
            }
            mlflow.log_param("algorithm", name)
            mlflow.log_param("cv_scoring", "f1")
            mlflow.log_param("best_params", json.dumps(search.best_params_, default=str))
            mlflow.log_param("smote_k_neighbors", smote_k)
            mlflow.log_metrics(metrics)
            for k, v in search.best_params_.items():
                # Store simple hyperparameter keys for easy UI comparison.
                mlflow.log_param(k.replace("model__", ""), str(v))
            cm_path = ARTIFACTS / f"confusion_matrix_{name}.png"
            ConfusionMatrixDisplay.from_predictions(y_test, pred, values_format="d")
            plt.title(f"Confusion Matrix — {name}")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            report_path = ARTIFACTS / f"classification_report_{name}.txt"
            report_path.write_text(classification_report(y_test, pred, zero_division=0), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

            # Feature importance for tree-based models only.
            core = best.named_steps["model"]
            if hasattr(core, "feature_importances_"):
                try:
                    names = best.named_steps["preprocessor"].get_feature_names_out()
                    imp = pd.Series(core.feature_importances_, index=names).sort_values(ascending=False).head(15)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    imp.sort_values().plot(kind="barh", ax=ax)
                    ax.set_title(f"Top Feature Importances — {name}")
                    fig.tight_layout()
                    fp = ARTIFACTS / f"feature_importance_{name}.png"
                    fig.savefig(fp, dpi=150)
                    plt.close(fig)
                    mlflow.log_artifact(str(fp), artifact_path="evaluation")
                except Exception as exc:
                    print("Feature importance plot skipped:", exc)

            signature = infer_signature(X_train.head(20), best.predict(X_train.head(20)))
            mlflow.sklearn.log_model(
                sk_model=best,
                artifact_path="model",
                serialization_format="cloudpickle",
                signature=signature,
                input_example=X_train.head(3),
            )
            result = {"algorithm": name, **metrics, "best_params": json.dumps(search.best_params_, default=str), "run_id": run.info.run_id}
            results.append(result)
            fitted_models.append((metrics["f1"], name, best, run.info.run_id))
            print(metrics)

    results_df = pd.DataFrame(results).sort_values(["f1", "recall"], ascending=False)
    results_df.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    print("\nModel comparison:\n", results_df[["algorithm", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))

    # Register the highest-F1 model (tie-break on recall). Registry support depends on MLflow backend.
    best_f1, best_name, best_model, best_run_id = sorted(fitted_models, key=lambda x: (x[0], x[2].named_steps["model"].__class__.__name__), reverse=True)[0]
    joblib.dump(best_model, ARTIFACTS / "model.pkl")
    print(f"\nSaved best pipeline: {best_name}, F1={best_f1:.4f}")
    try:
        model_uri = f"runs:/{best_run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print(f"Registered model version {registered.version} as {MODEL_NAME}")
        # Try to set Production stage where supported in this MLflow version.
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.transition_model_version_stage(
                name=MODEL_NAME, version=registered.version, stage="Production", archive_existing_versions=True
            )
            print("Transitioned model to Production stage.")
        except Exception as exc:
            print("Registry created, but Production stage transition may not be supported by this MLflow backend:", exc)
    except Exception as exc:
        print("Model registry step failed. Check MLflow backend/registry support:", exc)
        print("Best model is still saved to artifacts/model.pkl")

if __name__ == "__main__":
    main()
