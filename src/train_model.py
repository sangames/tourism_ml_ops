"""Tune, evaluate, and save the tourism package purchase classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "artifacts" / "processed"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORT_DIR = ROOT / "reports"
TARGET = "ProdTaken"


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing and classifier as one serializable, leakage-safe pipeline."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    classifier = RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=1
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def train_and_evaluate(train_path: str | Path = PROCESSED_DIR / "train.csv",
                       test_path: str | Path = PROCESSED_DIR / "test.csv",
                       model_path: str | Path = MODEL_PATH,
                       report_dir: str | Path = REPORT_DIR):
    """Tune on training folds only, evaluate once on holdout data, and save artifacts."""
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=TARGET), train[TARGET].astype(int)
    X_test, y_test = test.drop(columns=TARGET), test[TARGET].astype(int)
    pipeline = build_pipeline(X_train)
    parameter_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 12],
        "classifier__min_samples_leaf": [1, 3],
    }
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=folds,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        pre_dispatch="2*n_jobs",
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "positive_rate_test": float(y_test.mean()),
        "decision_threshold": 0.5,
    }
    model_path, report_dir = Path(model_path), Path(report_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (report_dir / "best_params.json").write_text(json.dumps(search.best_params_, indent=2))
    pd.DataFrame(search.cv_results_).to_csv(report_dir / "cv_results.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"],
    ).to_csv(report_dir / "confusion_matrix.csv")
    (report_dir / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0)
    )
    print("Best cross-validation ROC-AUC:", round(float(search.best_score_), 4))
    print("Best tuned parameters:")
    print(json.dumps(search.best_params_, indent=2))
    print("Holdout metrics:")
    print(json.dumps(metrics, indent=2))
    print("Holdout classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Holdout confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Saved model: {model_path}")
    print(f"Saved experiment records: {report_dir}")
    return best_model, metrics, search


if __name__ == "__main__":
    train_and_evaluate()
