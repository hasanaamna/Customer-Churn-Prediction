import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. CLEAN DATA
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ============================================================
# 3. REMOVE CUSTOMER ID
# ============================================================

df = df.drop(columns=["customerID"])


# ============================================================
# 4. FEATURES / TARGET
# ============================================================

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 5. COLUMN TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 8. MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 9. TRAIN
# ============================================================

pipeline.fit(X_train, y_train)


# ============================================================
# 10. PREDICT PROBABILITIES
# ============================================================

y_probability = pipeline.predict_proba(X_test)[:, 1]


print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"ROC-AUC: {roc_auc_score(y_test, y_probability):.4f}"
)


# ============================================================
# 11. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = np.arange(
    0.30,
    0.71,
    0.05
)


results = []


for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))


# ============================================================
# 12. BEST F1 THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n")
print("=" * 70)
print("BEST THRESHOLD BY F1")
print("=" * 70)

print(best_row)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

best_threshold = best_row["Threshold"]

best_predictions = (
    y_probability >= best_threshold
).astype(int)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)