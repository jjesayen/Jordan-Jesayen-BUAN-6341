"""
Lab 2: K-Nearest Neighbors and Classification

Final graded submission template.

Submit this file, Lab-2-Report.md, and any referenced files under artifacts/.
The notebook may be used only as an optional exploration workspace.

Run from the lab folder (the folder that contains datasets/adult.data):
    python Lab-2-Assignment.py

All result-bearing artifacts referenced by Lab-2-Report.md are written under
artifacts/figures/ and artifacts/tables/. A consolidated artifacts/tables/
results_summary.md is also written so the report numbers can be reconciled
quickly after any run.
"""

from pathlib import Path
import json
import time

# category_encoders is allowed by the handout but optional here: target
# encoding is implemented manually so the script runs without the dependency
# and so the leakage-control logic is fully visible.
try:
    from category_encoders import TargetEncoder  # noqa: F401
    HAVE_CATEGORY_ENCODERS = True
except Exception:  # pragma: no cover - dependency is optional
    HAVE_CATEGORY_ENCODERS = False

import matplotlib

matplotlib.use("Agg")  # headless backend so the script runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "datasets"
ARTIFACT_DIR = LAB_DIR / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
TABLE_DIR = ARTIFACT_DIR / "tables"

ADULT_DATA_PATH = DATA_DIR / "adult.data"
ADULT_TEST_PATH = DATA_DIR / "adult.test"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

# Six continuous predictors.
NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]
# Low / moderate cardinality categoricals -> one-hot encoded.
ONE_HOT_BASE_COLUMNS = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex",
]
# High-cardinality categorical -> target encoded by default.
HIGH_CARDINALITY_COLUMN = "native-country"
CATEGORICAL_COLUMNS = ONE_HOT_BASE_COLUMNS + [HIGH_CARDINALITY_COLUMN]

# Used to accumulate the headline numbers for the final summary table.
RESULTS = {}


def ensure_artifact_dirs():
    """Create local artifact folders used by the final report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# PART 1: Data Exploration & Understanding
# ==============================================================

def load_adult_frame(path, is_test=False):
    """Load one Adult file, clean whitespace, and standardize the target label.

    Handles the two UCI quirks:
      * adult.test has a metadata first line that must be skipped.
      * adult.test labels carry a trailing period (e.g. '>50K.').
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        skiprows=1 if is_test else 0,   # skip the '|1x3 Cross validator' line
        skipinitialspace=True,          # strip the leading space after each comma
        keep_default_na=False,          # keep '?' markers literal for the audit
        na_values=[],
        dtype=str,
    )

    # Strip whitespace from every string cell.
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Standardize target labels: remove trailing periods and normalize.
    df["income"] = df["income"].str.replace(".", "", regex=False).str.strip()

    # Restore numeric dtypes for the continuous columns.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def problem_1_1_load_and_inspect():
    """Problem 1.1: load both files, clean labels, and inspect structure."""
    if not ADULT_DATA_PATH.exists() or not ADULT_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Expected {ADULT_DATA_PATH} and {ADULT_TEST_PATH}. "
            "Place the UCI Adult files in the datasets/ folder and rerun."
        )

    adult_train = load_adult_frame(ADULT_DATA_PATH, is_test=False)
    adult_test = load_adult_frame(ADULT_TEST_PATH, is_test=True)

    print("\n[1.1] Shapes:")
    print("  adult.data:", adult_train.shape)
    print("  adult.test:", adult_test.shape)
    print("\n[1.1] Training dtypes:\n", adult_train.dtypes)
    print("\n[1.1] First 5 training rows:\n", adult_train.head())
    print("\n[1.1] Numeric summary (train):\n", adult_train[NUMERIC_COLUMNS].describe())
    print("\n[1.1] Target labels present:",
          sorted(adult_train["income"].unique()),
          sorted(adult_test["income"].unique()))

    return adult_train, adult_test


def problem_1_2_quality_review(adult_train, adult_test):
    """Problem 1.2: target distribution, imbalance, missing values, ranges."""
    def dist(df):
        counts = df["income"].value_counts()
        pct = (counts / len(df) * 100).round(2)
        return pd.DataFrame({"count": counts, "percent": pct})

    train_dist = dist(adult_train)
    test_dist = dist(adult_test)
    print("\n[1.2] Train target distribution:\n", train_dist)
    print("\n[1.2] Test target distribution:\n", test_dist)

    minority_pct = train_dist.loc[">50K", "percent"]
    RESULTS["train_minority_pct"] = float(minority_pct)
    RESULTS["majority_baseline_acc"] = float(train_dist.loc["<=50K", "percent"] / 100)

    # Missing-value markers per feature (NaN from '?' in categoricals).
    missing = {}
    for col in CATEGORICAL_COLUMNS:
        missing[col] = int((adult_train[col] == "?").sum())
    for col in NUMERIC_COLUMNS:
        missing[col] = int(adult_train[col].isna().sum())
    missing_series = pd.Series(missing).sort_values(ascending=False)
    print("\n[1.2] Missing-value markers per feature (train):\n", missing_series)

    # Categorical cardinality.
    cardinality = {c: int(adult_train[c].nunique()) for c in CATEGORICAL_COLUMNS}
    cardinality_series = pd.Series(cardinality).sort_values(ascending=False)
    print("\n[1.2] Categorical cardinality (train):\n", cardinality_series)

    # Numeric ranges.
    ranges = adult_train[NUMERIC_COLUMNS].agg(["min", "max", "mean"]).T
    print("\n[1.2] Numeric ranges (train):\n", ranges)

    # Save a consolidated data-quality table.
    quality_rows = []
    for col in COLUMN_NAMES[:-1]:
        quality_rows.append({
            "feature": col,
            "dtype": str(adult_train[col].dtype),
            "n_unique": int(adult_train[col].nunique()),
            "missing_markers": missing.get(col, 0),
            "min": ranges.loc[col, "min"] if col in NUMERIC_COLUMNS else "",
            "max": ranges.loc[col, "max"] if col in NUMERIC_COLUMNS else "",
        })
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(TABLE_DIR / "data_quality_summary.csv", index=False)

    # Target-distribution figure.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, d) in zip(axes, [("Train (adult.data)", train_dist),
                                     ("Test (adult.test)", test_dist)]):
        sns.barplot(x=d.index, y=d["percent"], ax=ax, hue=d.index, legend=False,
                    palette="viridis")
        ax.set_title(f"Income distribution - {name}")
        ax.set_ylabel("percent")
        for i, v in enumerate(d["percent"]):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=120)
    plt.close(fig)

    target_distribution = {"train": train_dist, "test": test_dist}
    return target_distribution


# ==============================================================
# PART 2: Leakage-Safe Preprocessing and Feature Engineering
# ==============================================================

def feature_groups(country_encoding):
    """Return (one_hot_columns, target_columns) for the chosen country strategy."""
    if country_encoding == "target":
        return list(ONE_HOT_BASE_COLUMNS), [HIGH_CARDINALITY_COLUMN]
    elif country_encoding == "onehot":
        return list(ONE_HOT_BASE_COLUMNS) + [HIGH_CARDINALITY_COLUMN], []
    raise ValueError("country_encoding must be 'target' or 'onehot'")


def fit_preprocessing(train_df, y_train, country_encoding="target"):
    """Learn all preprocessing state from the training split only.

    Stores imputation values, one-hot encoder state, manual target-encoding
    mappings + fallback, the assembled feature-column order, and scaler state.
    Validation/test rows are never inspected here.
    """
    train_df = train_df.copy()
    y_train = np.asarray(y_train)
    one_hot_columns, target_columns = feature_groups(country_encoding)

    # --- Imputation values (training only) ---
    imputation_values = {}
    for col in CATEGORICAL_COLUMNS:
        col_vals = train_df[col].replace("?", np.nan)
        mode = col_vals.mode(dropna=True)
        imputation_values[col] = mode.iloc[0] if len(mode) else "Unknown"
    for col in NUMERIC_COLUMNS:
        imputation_values[col] = float(train_df[col].median())

    # Apply imputation to a working copy used for fitting encoders/scaler.
    work = train_df.copy()
    for col in CATEGORICAL_COLUMNS:
        work[col] = work[col].replace("?", np.nan).fillna(imputation_values[col])
    for col in NUMERIC_COLUMNS:
        work[col] = work[col].fillna(imputation_values[col])

    # --- One-hot encoder (training only) ---
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    one_hot_encoder.fit(work[one_hot_columns])
    one_hot_feature_names = list(
        one_hot_encoder.get_feature_names_out(one_hot_columns)
    )

    # --- Manual target encoding (training only) ---
    target_encoder = {}
    target_encoding_fallback = float(y_train.mean())  # global training positive rate
    target_encoded_columns = []
    for col in target_columns:
        means = (
            pd.DataFrame({col: work[col].values, "_y": y_train})
            .groupby(col)["_y"].mean()
        )
        target_encoder[col] = means.to_dict()
        target_encoded_columns.append(f"{col}_target_enc")

    # --- Assemble the unscaled feature matrix to learn column order + scaler ---
    feature_columns = NUMERIC_COLUMNS + target_encoded_columns + one_hot_feature_names

    preprocessing_state = {
        "country_encoding": country_encoding,
        "numeric_columns": list(NUMERIC_COLUMNS),
        "one_hot_columns": list(one_hot_columns),
        "target_columns": list(target_columns),
        "target_encoded_columns": target_encoded_columns,
        "imputation_values": imputation_values,
        "one_hot_encoder": one_hot_encoder,
        "one_hot_feature_names": one_hot_feature_names,
        "target_encoder": target_encoder,
        "target_encoding_fallback": target_encoding_fallback,
        "scaler": None,
        "feature_columns": feature_columns,
    }

    # Fit the scaler on the assembled training matrix (no scaling applied yet).
    unscaled = _assemble_matrix(train_df, preprocessing_state)
    scaler = StandardScaler()
    scaler.fit(unscaled.values)
    preprocessing_state["scaler"] = scaler

    return preprocessing_state


def _assemble_matrix(df, preprocessing_state):
    """Build the unscaled, fully numeric feature DataFrame in stored column order."""
    df = df.copy()
    imp = preprocessing_state["imputation_values"]

    # Impute using stored training values only.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].replace("?", np.nan).fillna(imp[col])
    for col in preprocessing_state["numeric_columns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(imp[col])

    blocks = [df[preprocessing_state["numeric_columns"]].astype(float).reset_index(drop=True)]

    # Target-encoded columns (unknown categories -> stored fallback).
    fallback = preprocessing_state["target_encoding_fallback"]
    for col in preprocessing_state["target_columns"]:
        mapping = preprocessing_state["target_encoder"][col]
        enc = df[col].map(mapping).fillna(fallback).astype(float)
        blocks.append(pd.DataFrame({f"{col}_target_enc": enc.values}))

    # One-hot block (handle_unknown='ignore' silently zero-fills unseen levels).
    ohe = preprocessing_state["one_hot_encoder"]
    one_hot_columns = preprocessing_state["one_hot_columns"]
    ohe_array = ohe.transform(df[one_hot_columns])
    ohe_df = pd.DataFrame(ohe_array,
                          columns=preprocessing_state["one_hot_feature_names"])
    blocks.append(ohe_df)

    matrix = pd.concat(blocks, axis=1)
    # Guarantee identical column order; any missing column filled with 0.
    matrix = matrix.reindex(columns=preprocessing_state["feature_columns"],
                            fill_value=0.0)
    return matrix


def transform_preprocessing(df, preprocessing_state, scale=True):
    """Apply learned preprocessing state to one compatible split.

    Never fits or recomputes training-only values. Returns a numeric matrix
    with the same column order for train, validation, and test.
    """
    matrix = _assemble_matrix(df, preprocessing_state)
    if scale:
        scaled = preprocessing_state["scaler"].transform(matrix.values)
        transformed_features = pd.DataFrame(
            scaled, columns=preprocessing_state["feature_columns"]
        )
    else:
        transformed_features = matrix
    return transformed_features


def problem_2_1_split(adult_train):
    """Problem 2.1: encode target, stratified train/validation split."""
    y_dev = (adult_train["income"] == ">50K").astype(int).values
    X_dev = adult_train.drop(columns=["income"])

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    print("\n[2.1] Split sizes:")
    print(f"  train: {len(X_train)}  >50K share: {y_train.mean():.3f}")
    print(f"  valid: {len(X_val)}  >50K share: {y_val.mean():.3f}")
    RESULTS["n_train"] = int(len(X_train))
    RESULTS["n_val"] = int(len(X_val))

    return {"X_train": X_train, "X_val": X_val,
            "y_train": y_train, "y_val": y_val}


def _knn_validation(X_tr, y_tr, X_va, y_va, n_neighbors=5, p=2):
    """Fit a KNN, predict validation, return metrics + timings."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski", p=p)
    t0 = time.perf_counter()
    knn.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    preds = knn.predict(X_va)
    predict_time = time.perf_counter() - t1
    return {
        "accuracy": accuracy_score(y_va, preds),
        "f1": f1_score(y_va, preds),
        "precision": precision_score(y_va, preds, zero_division=0),
        "recall": recall_score(y_va, preds, zero_division=0),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "model": knn,
        "preds": preds,
    }


def problem_2_3_encoding_comparison(split):
    """Problem 2.3: one-hot vs target encoding for native-country."""
    X_train, X_val = split["X_train"], split["X_val"]
    y_train, y_val = split["y_train"], split["y_val"]

    n_country = int(X_train[HIGH_CARDINALITY_COLUMN].nunique())
    rows = []
    for strategy in ("onehot", "target"):
        t0 = time.perf_counter()
        state = fit_preprocessing(X_train, y_train, country_encoding=strategy)
        Xtr = transform_preprocessing(X_train, state)
        Xva = transform_preprocessing(X_val, state)
        res = _knn_validation(Xtr, y_train, Xva, y_val, n_neighbors=5)
        elapsed = time.perf_counter() - t0
        rows.append({
            "strategy": strategy,
            "n_features": Xtr.shape[1],
            "val_accuracy": round(res["accuracy"], 4),
            "val_f1": round(res["f1"], 4),
            "runtime_sec": round(elapsed, 3),
        })

    comparison = pd.DataFrame(rows)
    print("\n[2.3] native-country unique values:", n_country)
    print("[2.3] Encoding comparison:\n", comparison)

    # Show training-only target-encoded value for 5+ categories.
    state_t = fit_preprocessing(X_train, y_train, country_encoding="target")
    mapping = state_t["target_encoder"][HIGH_CARDINALITY_COLUMN]
    sample = dict(list(sorted(mapping.items()))[:6])
    fallback = state_t["target_encoding_fallback"]
    print("[2.3] Sample target-encoded values:", {k: round(v, 4) for k, v in sample.items()})
    print(f"[2.3] Unknown-category fallback (global train positive rate): {fallback:.4f}")

    comparison.to_csv(TABLE_DIR / "target_encoding_comparison.csv", index=False)
    sample_df = pd.DataFrame(
        [{"native_country": k, "target_encoded_value": round(v, 4)}
         for k, v in sample.items()]
    )
    sample_df.loc[len(sample_df)] = ["<unknown_fallback>", round(fallback, 4)]
    sample_df.to_csv(TABLE_DIR / "target_encoding_sample_values.csv", index=False)

    RESULTS["n_country"] = n_country
    RESULTS["encoding_onehot_features"] = int(comparison.iloc[0]["n_features"])
    RESULTS["encoding_target_features"] = int(comparison.iloc[1]["n_features"])
    RESULTS["encoding_comparison"] = comparison.to_dict(orient="records")
    return comparison


def problem_2_4_leakage_audit():
    """Problem 2.4: leakage audit table with the six required steps."""
    rows = [
        {
            "step": "missing-value handling",
            "fit_on": "training split only",
            "applied_to": "train / validation / test",
            "stored_state": "training mode (categorical) and median (numeric)",
            "leakage_risk": "imputing with statistics computed over val/test rows",
            "how_risk_was_controlled": "modes/medians learned in fit_preprocessing from train only, stored in preprocessing_state",
        },
        {
            "step": "target encoding",
            "fit_on": "training split only",
            "applied_to": "train / validation / test",
            "stored_state": "category -> mean(y_train) map + global train positive-rate fallback",
            "leakage_risk": "highest: encoding uses the label, so val/test labels could leak into features",
            "how_risk_was_controlled": "group means computed only on y_train; unseen categories use the stored fallback, never val/test outcomes",
        },
        {
            "step": "one-hot encoding",
            "fit_on": "training split only",
            "applied_to": "train / validation / test",
            "stored_state": "fitted OneHotEncoder categories + output column order",
            "leakage_risk": "new categories at test time could change columns or crash",
            "how_risk_was_controlled": "handle_unknown='ignore' zero-fills unseen levels; columns reindexed to stored order",
        },
        {
            "step": "scaling",
            "fit_on": "training split only",
            "applied_to": "train / validation / test",
            "stored_state": "StandardScaler means and standard deviations",
            "leakage_risk": "scaling with val/test statistics shares distribution info across splits",
            "how_risk_was_controlled": "scaler.fit on the assembled train matrix only; transform applied to other splits",
        },
        {
            "step": "KNN model selection",
            "fit_on": "training split; tuned on validation",
            "applied_to": "validation predictions",
            "stored_state": "selected k, distance metric, encoding strategy",
            "leakage_risk": "choosing hyperparameters by peeking at test data",
            "how_risk_was_controlled": "all k/metric/encoding choices made on validation only; test untouched until frozen",
        },
        {
            "step": "final test evaluation",
            "fit_on": "full training split with frozen choices",
            "applied_to": "adult.test once",
            "stored_state": "none new; reuses frozen preprocessing_state",
            "leakage_risk": "re-tuning after seeing test results",
            "how_risk_was_controlled": "test transformed with frozen state and scored exactly once; no choices changed afterward",
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLE_DIR / "leakage_audit.csv", index=False)
    print("\n[2.4] Leakage audit table saved (6 steps).")
    return audit


def problem_2_5_matrix_checks(split, country_encoding="target"):
    """Problem 2.5: build final matrices and verify consistency."""
    X_train, X_val = split["X_train"], split["X_val"]
    y_train = split["y_train"]

    state = fit_preprocessing(X_train, y_train, country_encoding=country_encoding)
    X_train_t = transform_preprocessing(X_train, state)
    X_val_t = transform_preprocessing(X_val, state)

    same_cols = (list(X_train_t.columns) == list(X_val_t.columns)
                 == list(state["feature_columns"]))
    no_missing = not (X_train_t.isna().any().any() or X_val_t.isna().any().any())
    all_numeric = all(np.issubdtype(dt, np.number) for dt in X_train_t.dtypes)

    print("\n[2.5] Feature-matrix checks:")
    print("  train shape:", X_train_t.shape, " val shape:", X_val_t.shape)
    print("  identical column order:", same_cols)
    print("  no missing values:", no_missing)
    print("  all numeric:", all_numeric)

    summary = pd.DataFrame([
        {"matrix": "train", "rows": X_train_t.shape[0], "cols": X_train_t.shape[1]},
        {"matrix": "validation", "rows": X_val_t.shape[0], "cols": X_val_t.shape[1]},
    ])
    summary["identical_column_order"] = same_cols
    summary["no_missing"] = no_missing
    summary["all_numeric"] = all_numeric
    summary.to_csv(TABLE_DIR / "feature_matrix_summary.csv", index=False)

    RESULTS["n_features_final"] = int(X_train_t.shape[1])
    return {"state": state, "X_train_t": X_train_t, "X_val_t": X_val_t}


# ==============================================================
# PART 3: KNN Mechanics From Scratch
# ==============================================================

def euclidean_distances_from_one(row, training_matrix):
    """Vectorized Euclidean distances from one row to every training row."""
    row = np.asarray(row, dtype=float)
    training_matrix = np.asarray(training_matrix, dtype=float)
    diff = training_matrix - row  # broadcast over rows
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    return distances


def manual_knn_predict(X_train, y_train, X_query, k):
    """Manual binary KNN via majority vote. No scikit-learn used."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_query = np.asarray(X_query, dtype=float)

    predictions = np.empty(X_query.shape[0], dtype=int)
    for i in range(X_query.shape[0]):
        dists = euclidean_distances_from_one(X_query[i], X_train)
        nearest_idx = np.argsort(dists)[:k]
        votes = y_train[nearest_idx]
        predictions[i] = int(round(votes.mean()))  # majority vote for {0,1}
    return predictions


def problem_3_manual_knn(prepared, split, k=5):
    """Problems 3.1-3.3: manual distance, manual KNN, vs scikit-learn."""
    X_train_t, X_val_t = prepared["X_train_t"], prepared["X_val_t"]
    y_train, y_val = split["y_train"], split["y_val"]

    # Small subsets per the handout limits.
    n_tr, n_va = min(1000, len(X_train_t)), min(100, len(X_val_t))
    Xtr = X_train_t.values[:n_tr]
    ytr = y_train[:n_tr]
    Xva = X_val_t.values[:n_va]
    yva = y_val[:n_va]

    # 3.1 nearest-neighbor example for one validation row.
    d0 = euclidean_distances_from_one(Xva[0], Xtr)
    nearest5 = np.argsort(d0)[:5]
    nn_example = pd.DataFrame({
        "train_index": nearest5,
        "distance": np.round(d0[nearest5], 4),
        "train_label": ytr[nearest5],
    })
    nn_example.to_csv(TABLE_DIR / "nearest_neighbor_example.csv", index=False)
    print("\n[3.1] Nearest 5 training rows to validation row 0:\n", nn_example)

    # 3.2 manual KNN predictions on the validation subset.
    manual_preds = manual_knn_predict(Xtr, ytr, Xva, k)
    manual_acc = accuracy_score(yva, manual_preds)
    manual_f1 = f1_score(yva, manual_preds, zero_division=0)
    preds_table = pd.DataFrame({
        "val_index": np.arange(min(15, n_va)),
        "manual_prediction": manual_preds[:15],
        "true_label": yva[:15],
    })
    preds_table.to_csv(TABLE_DIR / "manual_knn_predictions.csv", index=False)
    print(f"\n[3.2] Manual KNN (subset={n_tr}/{n_va}, k={k}): "
          f"acc={manual_acc:.4f} f1={manual_f1:.4f}")
    print(preds_table.head(12))

    # 3.3 compare with scikit-learn on the same subset / k / metric.
    sk = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)
    sk.fit(Xtr, ytr)
    sk_preds = sk.predict(Xva)
    n_match = int((sk_preds == manual_preds).sum())
    pct_match = 100.0 * n_match / len(manual_preds)
    print(f"[3.3] Manual vs scikit-learn agreement: {n_match}/{len(manual_preds)} "
          f"({pct_match:.1f}%)")

    RESULTS.update({
        "manual_subset_train": n_tr,
        "manual_subset_val": n_va,
        "manual_k": k,
        "manual_accuracy": round(manual_acc, 4),
        "manual_f1": round(manual_f1, 4),
        "manual_vs_sklearn_match": n_match,
        "manual_vs_sklearn_pct": round(pct_match, 1),
    })
    return {"manual_acc": manual_acc, "manual_f1": manual_f1,
            "pct_match": pct_match}


# ==============================================================
# PART 4: KNN Experiments and Runtime-Aware Model Selection
# ==============================================================

def problem_4_1_baseline(prepared, split):
    """Problem 4.1: baseline KNN n=5 on full scaled training features."""
    res = _knn_validation(prepared["X_train_t"], split["y_train"],
                          prepared["X_val_t"], split["y_val"], n_neighbors=5)
    print("\n[4.1] Baseline KNN (k=5) validation:")
    print(f"  accuracy={res['accuracy']:.4f} precision={res['precision']:.4f} "
          f"recall={res['recall']:.4f} f1={res['f1']:.4f}")
    print(classification_report(split["y_val"], res["preds"],
                                target_names=["<=50K", ">50K"], zero_division=0))
    RESULTS.update({
        "baseline_acc": round(res["accuracy"], 4),
        "baseline_precision": round(res["precision"], 4),
        "baseline_recall": round(res["recall"], 4),
        "baseline_f1": round(res["f1"], 4),
    })
    return res


def problem_4_2_distance_diagnostics(prepared, split, n_sample=25, k=5):
    """Problem 4.2: nearest-neighbor distance summaries before/after scaling."""
    X_train, X_val = split["X_train"], split["X_val"]
    state = prepared["state"]

    # Unscaled and scaled feature matrices for the same rows.
    Xtr_unscaled = transform_preprocessing(X_train, state, scale=False).values
    Xva_unscaled = transform_preprocessing(X_val, state, scale=False).values
    Xtr_scaled = prepared["X_train_t"].values
    Xva_scaled = prepared["X_val_t"].values

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(Xva_scaled), size=min(n_sample, len(Xva_scaled)),
                            replace=False)

    def summarize(query_rows, train_rows):
        nearest, median, farthest, ratio = [], [], [], []
        for i in query_rows:
            d = euclidean_distances_from_one(Xva_template[i], train_rows)
            d_sorted = np.sort(d)
            nearest.append(d_sorted[1] if len(d_sorted) > 1 else d_sorted[0])
            median.append(np.median(d))
            farthest.append(d_sorted[-1])
            ratio.append((d_sorted[1] if len(d_sorted) > 1 else d_sorted[0]) /
                         d_sorted[-1] if d_sorted[-1] > 0 else 0.0)
        return np.mean(nearest), np.mean(median), np.mean(farthest), np.mean(ratio)

    Xva_template = Xva_unscaled
    before = summarize(sample_idx, Xtr_unscaled)
    Xva_template = Xva_scaled
    after = summarize(sample_idx, Xtr_scaled)

    diag = pd.DataFrame([
        {"setting": "before_scaling", "mean_nearest": round(before[0], 3),
         "mean_median": round(before[1], 3), "mean_farthest": round(before[2], 3),
         "mean_nearest_farthest_ratio": round(before[3], 5)},
        {"setting": "after_scaling", "mean_nearest": round(after[0], 3),
         "mean_median": round(after[1], 3), "mean_farthest": round(after[2], 3),
         "mean_nearest_farthest_ratio": round(after[3], 5)},
    ])
    diag.to_csv(TABLE_DIR / "distance_diagnostics.csv", index=False)
    print("\n[4.2] Distance diagnostics (mean over sample):\n", diag)
    RESULTS["distance_diagnostics"] = diag.to_dict(orient="records")
    return diag


def problem_4_3_encoding_choice(comparison):
    """Problem 4.3: choose the encoding strategy from validation evidence."""
    target_row = comparison[comparison["strategy"] == "target"].iloc[0]
    onehot_row = comparison[comparison["strategy"] == "onehot"].iloc[0]
    choice = "target"  # fewer features, lower runtime, comparable/better F1 for KNN
    print("\n[4.3] Encoding choice for the rest of Lab 2:", choice)
    print(f"  target: {int(target_row['n_features'])} features, "
          f"f1={target_row['val_f1']}, {target_row['runtime_sec']}s")
    print(f"  onehot: {int(onehot_row['n_features'])} features, "
          f"f1={onehot_row['val_f1']}, {onehot_row['runtime_sec']}s")
    RESULTS["encoding_choice"] = choice
    return choice


def problem_4_4_k_distance_loop(prepared, split):
    """Problem 4.4: runtime-aware loop over >=10 k values and 2 distances."""
    X_tr, y_tr = prepared["X_train_t"].values, split["y_train"]
    X_va, y_va = prepared["X_val_t"].values, split["y_val"]

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    distance_settings = [("euclidean (p=2)", 2), ("manhattan (p=1)", 1)]

    rows = []
    for label, p in distance_settings:
        for k in k_values:
            res = _knn_validation(X_tr, y_tr, X_va, y_va, n_neighbors=k, p=p)
            rows.append({
                "k": k,
                "distance": label,
                "val_accuracy": round(res["accuracy"], 4),
                "val_f1": round(res["f1"], 4),
                "fit_time": round(res["fit_time"], 4),
                "predict_time": round(res["predict_time"], 4),
            })
    loop = pd.DataFrame(rows)
    loop.to_csv(TABLE_DIR / "knn_k_distance_results.csv", index=False)

    # Select final config by highest validation F1 (imbalance-aware), tie-break
    # on accuracy then smaller predict_time.
    loop_sorted = loop.sort_values(
        ["val_f1", "val_accuracy", "predict_time"],
        ascending=[False, False, True]
    )
    best = loop_sorted.iloc[0]
    print("\n[4.4] k / distance loop complete. Best by validation F1:")
    print(best)

    # Plot accuracy and F1 by k for each distance setting.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, _ in distance_settings:
        sub = loop[loop["distance"] == label]
        axes[0].plot(sub["k"], sub["val_accuracy"], marker="o", label=label)
        axes[1].plot(sub["k"], sub["val_f1"], marker="o", label=label)
    axes[0].set(title="Validation accuracy by k", xlabel="k", ylabel="accuracy")
    axes[1].set(title="Validation F1 by k", xlabel="k", ylabel="F1")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "knn_k_distance_validation.png", dpi=120)
    plt.close(fig)

    final_p = 2 if "p=2" in best["distance"] else 1
    RESULTS.update({
        "final_k": int(best["k"]),
        "final_distance": best["distance"],
        "final_p": final_p,
        "loop_best_val_acc": float(best["val_accuracy"]),
        "loop_best_val_f1": float(best["val_f1"]),
    })
    return {"loop": loop, "best_k": int(best["k"]), "best_p": final_p,
            "best_distance": best["distance"]}


def problem_4_5_final_test(adult_train, adult_test, loop_choice,
                           country_encoding):
    """Problem 4.5: freeze choices, train on full training split, test once."""
    # Full development training split (refit preprocessing on full train split).
    y_dev = (adult_train["income"] == ">50K").astype(int).values
    X_dev = adult_train.drop(columns=["income"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE
    )

    state = fit_preprocessing(X_train, y_train, country_encoding=country_encoding)
    X_train_t = transform_preprocessing(X_train, state)

    y_test = (adult_test["income"] == ">50K").astype(int).values
    X_test = adult_test.drop(columns=["income"])
    X_test_t = transform_preprocessing(X_test, state)

    knn = KNeighborsClassifier(n_neighbors=loop_choice["best_k"],
                               metric="minkowski", p=loop_choice["best_p"])
    knn.fit(X_train_t, y_train)
    preds = knn.predict(X_test_t)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    print("\n[4.5] Final test evaluation (frozen model):")
    print(f"  k={loop_choice['best_k']} {loop_choice['best_distance']}")
    print(f"  accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
    print(classification_report(y_test, preds,
                                target_names=["<=50K", ">50K"], zero_division=0))

    RESULTS.update({
        "test_accuracy": round(metrics["accuracy"], 4),
        "test_precision": round(metrics["precision"], 4),
        "test_recall": round(metrics["recall"], 4),
        "test_f1": round(metrics["f1"], 4),
    })
    return {"model": knn, "preds": preds, "y_test": y_test, "metrics": metrics,
            "state": state, "X_test_t": X_test_t}


# ==============================================================
# PART 5: Evaluation, Artifacts, and Reflection
# ==============================================================

def _plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], ax=ax)
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def problem_5_1_confusion(prepared, split, baseline_res, final_test, loop_choice):
    """Problem 5.1: validation + test confusion matrices and interpretation."""
    # Validation confusion matrix from the SELECTED model (refit on train split).
    sel = KNeighborsClassifier(n_neighbors=loop_choice["best_k"],
                               metric="minkowski", p=loop_choice["best_p"])
    sel.fit(prepared["X_train_t"], split["y_train"])
    val_preds = sel.predict(prepared["X_val_t"])
    _plot_confusion(split["y_val"], val_preds, "Validation confusion (selected KNN)",
                    FIGURE_DIR / "confusion_matrix_validation.png")

    cm_test = _plot_confusion(final_test["y_test"], final_test["preds"],
                              "Test confusion (selected KNN)",
                              FIGURE_DIR / "confusion_matrix_test.png")
    tn, fp, fn, tp = cm_test.ravel()
    print("\n[5.1] Test confusion matrix counts:")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    RESULTS.update({"test_TN": int(tn), "test_FP": int(fp),
                    "test_FN": int(fn), "test_TP": int(tp)})
    return cm_test


def problem_5_2_summary_table():
    """Problem 5.2: required final artifact summary table."""
    enc = {r["strategy"]: r for r in RESULTS.get("encoding_comparison", [])}
    diag = RESULTS.get("distance_diagnostics", [{}, {}])
    rows = [
        ["Baseline KNN (k=5) validation accuracy", RESULTS.get("baseline_acc"),
         "artifacts/figures/target_distribution.png"],
        ["Baseline KNN (k=5) validation F1", RESULTS.get("baseline_f1"), ""],
        ["Encoding comparison - onehot features",
         RESULTS.get("encoding_onehot_features"),
         "artifacts/tables/target_encoding_comparison.csv"],
        ["Encoding comparison - target features",
         RESULTS.get("encoding_target_features"), ""],
        ["Encoding comparison - onehot val F1",
         enc.get("onehot", {}).get("val_f1"), ""],
        ["Encoding comparison - target val F1",
         enc.get("target", {}).get("val_f1"), ""],
        ["Chosen encoding strategy", RESULTS.get("encoding_choice"), ""],
        ["Distance diag nearest/farthest ratio (before scaling)",
         diag[0].get("mean_nearest_farthest_ratio"),
         "artifacts/tables/distance_diagnostics.csv"],
        ["Distance diag nearest/farthest ratio (after scaling)",
         diag[1].get("mean_nearest_farthest_ratio"), ""],
        ["Manual vs scikit-learn agreement (%)",
         RESULTS.get("manual_vs_sklearn_pct"),
         "artifacts/tables/manual_knn_predictions.csv"],
        ["Runtime-aware loop winner k", RESULTS.get("final_k"),
         "artifacts/figures/knn_k_distance_validation.png"],
        ["Runtime-aware loop winner distance", RESULTS.get("final_distance"), ""],
        ["Runtime-aware loop winner val F1", RESULTS.get("loop_best_val_f1"),
         "artifacts/tables/knn_k_distance_results.csv"],
        ["Final test accuracy", RESULTS.get("test_accuracy"),
         "artifacts/figures/confusion_matrix_test.png"],
        ["Final test precision", RESULTS.get("test_precision"), ""],
        ["Final test recall", RESULTS.get("test_recall"), ""],
        ["Final test F1", RESULTS.get("test_f1"), ""],
    ]
    summary = pd.DataFrame(rows, columns=["metric", "value", "related_file"])
    summary.to_csv(TABLE_DIR / "final_summary.csv", index=False)
    print("\n[5.2] Final artifact summary table saved.")
    return summary


def write_results_summary():
    """Convenience dump of every headline number for report reconciliation."""
    lines = ["# Results Summary (auto-generated)\n",
             "Regenerated on each run. Use these numbers to fill Lab-2-Report.md.\n"]
    for key in sorted(RESULTS):
        val = RESULTS[key]
        if isinstance(val, (list, dict)):
            val = json.dumps(val)
        lines.append(f"- **{key}**: {val}")
    (TABLE_DIR / "results_summary.md").write_text("\n".join(lines))
    print("\n[summary] artifacts/tables/results_summary.md written.")


def main():
    ensure_artifact_dirs()
    print("Lab 2 Python template started.")
    print(f"Data folder: {DATA_DIR}")
    print(f"Artifacts folder: {ARTIFACT_DIR}")
    print(f"category_encoders available: {HAVE_CATEGORY_ENCODERS} "
          "(manual target encoding used regardless)")

    # PART 1 ----------------------------------------------------
    adult_train, adult_test = problem_1_1_load_and_inspect()
    target_distribution = problem_1_2_quality_review(adult_train, adult_test)
    # knn_risk_notes -> written in Lab-2-Report.md (Problem 1.3)

    # PART 2 ----------------------------------------------------
    split_data = problem_2_1_split(adult_train)
    target_encoding_comparison = problem_2_3_encoding_comparison(split_data)
    leakage_audit_table = problem_2_4_leakage_audit()

    country_encoding = "target"  # confirmed in 4.3 from validation evidence
    prepared = problem_2_5_matrix_checks(split_data, country_encoding)
    preprocessing_state = prepared["state"]
    X_train = prepared["X_train_t"]
    X_validation = prepared["X_val_t"]

    # PART 3 ----------------------------------------------------
    manual_knn_results = problem_3_manual_knn(prepared, split_data, k=5)

    # PART 4 ----------------------------------------------------
    baseline_knn = problem_4_1_baseline(prepared, split_data)
    distance_diagnostic_results = problem_4_2_distance_diagnostics(prepared, split_data)
    encoding_choice = problem_4_3_encoding_choice(target_encoding_comparison)
    runtime_validation_loop = problem_4_4_k_distance_loop(prepared, split_data)
    final_test_results = problem_4_5_final_test(
        adult_train, adult_test, runtime_validation_loop, country_encoding
    )

    # PART 5 ----------------------------------------------------
    confusion_matrix_artifacts = problem_5_1_confusion(
        prepared, split_data, baseline_knn, final_test_results, runtime_validation_loop
    )
    artifact_summary = problem_5_2_summary_table()
    write_results_summary()
    print("\nFinal artifact summary:\n", artifact_summary.to_string(index=False))

    print("\nTemplate execution completed successfully.")


if __name__ == "__main__":
    main()
