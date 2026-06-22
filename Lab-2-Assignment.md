# Lab 2 Assignment Handout

> Final graded submissions must use `Lab-2-Assignment.py`, `Lab-2-Report.md`, and referenced files under `artifacts/`. The notebook may be used only as an optional exploration/pre-flow workspace. Do not submit a Jupyter notebook as the final lab submission.

# Lab 2: K-Nearest Neighbors and Leakage-Safe Classification

**Course:** Machine Learning (BUAN 6341)  
**Semester:** Summer 2026  
**Total Points:** 100

---

## Objective

In this lab, you will:
- Explore and preprocess the Adult Income dataset
- Build reusable preprocessing functions that separate fitting from transformation
- Apply one-hot encoding and target encoding without leaking validation or test information
- Scale numeric features for a distance-based model
- Implement a small KNN classifier manually to demonstrate the algorithm mechanics
- Train, tune, and evaluate scikit-learn KNN models
- Explain how distance metrics, `k`, scaling, encoding, runtime, and class imbalance affect KNN performance

This lab is scoped to **Module 2** plus the Module 1 data-preparation and feature-engineering skills needed to make Module 2 work, including target encoding for high-cardinality categorical features. Later labs cover decision trees, ROC/PR curves, cross-validation, advanced tuning, feature importance, and learning-curve diagnostics.

---

## Dataset Information

**Dataset:** Adult Income Dataset (Census Income)  
**Source:** UCI Machine Learning Repository  
**Task:** Binary classification - predict whether income exceeds $50K/year

**Files:**
- `adult.data` - Development data (32,561 instances)
- `adult.test` - Final test data (16,281 instances)
- `adult.names` - Dataset documentation

**Attributes (14 features + 1 target):**
1. `age`: continuous
2. `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
3. `fnlwgt`: continuous (final weight)
4. `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
5. `education-num`: continuous
6. `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
7. `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
8. `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
9. `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
10. `sex`: Female, Male
11. `capital-gain`: continuous
12. `capital-loss`: continuous
13. `hours-per-week`: continuous
14. `native-country`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, ...
15. `income`: <=50K, >50K (target variable)

**Important Notes:**
- Missing values are represented as ` ?` (space before question mark)
- The test file has a slightly different format, including a first metadata line and periods after target labels
- Use `random_state=42` for reproducibility throughout this lab
- Keep the provided test file isolated for final evaluation after you choose KNN settings on the validation split
- Do not use `sklearn.pipeline.Pipeline`, `make_pipeline`, `ColumnTransformer`, `GridSearchCV`, or cross-validation in this lab

---

## Required Design Pattern

Your final code must make the preprocessing workflow inspectable. Implement these functions in `Lab-2-Assignment.py`:

```python
def fit_preprocessing(train_df, y_train):
    """Learn all training-only preprocessing state."""
    return preprocessing_state


def transform_preprocessing(df, preprocessing_state):
    """Apply learned preprocessing state to any compatible split."""
    return X
```

`preprocessing_state` can be a dictionary or another explicit structure. It should contain the values needed to transform validation and test data consistently, such as imputation values, one-hot encoder state, target-encoding mappings, unknown-category fallback values, feature-column order, and scaler state.

---

## Table of Contents

### PART 1: Data Exploration & Understanding (10 points)
- **Problem 1.1:** Load, Clean Labels, and Inspect Files (4 points)
- **Problem 1.2:** Target Distribution and Data Quality Review (4 points)
- **Problem 1.3:** Baseline Risk Notes for KNN (2 points)

### PART 2: Leakage-Safe Preprocessing and Feature Engineering (30 points)
- **Problem 2.1:** Split Before Fitting Transformations (5 points)
- **Problem 2.2:** Implement `fit_preprocessing` and `transform_preprocessing` (10 points)
- **Problem 2.3:** Target Encoding Comparison for High-Cardinality Categories (8 points)
- **Problem 2.4:** Leakage Audit Table (4 points)
- **Problem 2.5:** Final Feature Matrix Checks (3 points)

### PART 3: KNN Mechanics From Scratch (20 points)
- **Problem 3.1:** Manual Euclidean Distance Function (5 points)
- **Problem 3.2:** Manual KNN Prediction on a Small Subset (8 points)
- **Problem 3.3:** Compare Manual KNN to scikit-learn KNN (4 points)
- **Problem 3.4:** Manual KNN Reflection (3 points)

### PART 4: KNN Experiments and Runtime-Aware Model Selection (25 points)
- **Problem 4.1:** Baseline scikit-learn KNN (5 points)
- **Problem 4.2:** Distance Diagnostics Before and After Scaling (5 points)
- **Problem 4.3:** Encoding Strategy Comparison (5 points)
- **Problem 4.4:** Runtime-Aware `k` and Distance-Metric Validation Loop (7 points)
- **Problem 4.5:** Final Test Evaluation (3 points)

### PART 5: Evaluation, Artifacts, and Reflection (15 points)
- **Problem 5.1:** Confusion Matrices and Metric Interpretation (5 points)
- **Problem 5.2:** Required Artifact Summary Table (5 points)
- **Problem 5.3:** Final Recommendation and Reflection (5 points)

---

**Total Points: 100**

---

## Import Required Libraries

Put required imports and setup code near the top of `Lab-2-Assignment.py`. Use the optional notebook only for exploration/pre-flow work.

Allowed model-selection style: manual train/validation loops only.
Not allowed in Lab 2: `GridSearchCV`, cross-validation, ROC/AUC analysis, decision trees, feature importance, learning curves, or scikit-learn pipeline abstractions.

---

# PART 1: Data Exploration & Understanding (10 points)

In this section, you will load and inspect the Adult Income dataset and identify issues that will matter for KNN.

## Problem 1.1: Load, Clean Labels, and Inspect Files (4 points)

**Task:**
1. Load both `adult.data` and `adult.test` into pandas DataFrames
2. Add the appropriate column names to both datasets
3. Handle the first non-data line in `adult.test`
4. Strip whitespace from string columns
5. Standardize target labels so train and test use `<=50K` and `>50K`
6. Display shapes, data types, the first 5 training rows, and a numeric summary

**Column Names (in order):**
```python
['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
 'hours-per-week', 'native-country', 'income']
```

## Problem 1.2: Target Distribution and Data Quality Review (4 points)

**Task:**
1. Check the target distribution in development and final test data
2. Calculate the percentage of each class
3. Identify class imbalance
4. Count missing-value markers by feature
5. Review categorical cardinality and numeric ranges
6. Save at least one target-distribution or data-quality artifact under `artifacts/`

**Hints:**
- Missing values may appear as `?` after whitespace stripping
- You do not need advanced ROC/PR analysis in Lab 2
- Focus on accuracy, precision, recall, F1, and confusion matrices

## Problem 1.3: Baseline Risk Notes for KNN (2 points)

**Task:**
Write 5-8 lines in `Lab-2-Report.md` explaining which data properties may make KNN difficult here. Address at least:
1. class imbalance
2. high-cardinality categorical features
3. large numeric ranges
4. runtime cost on a 32K-row development dataset

---

# PART 2: Leakage-Safe Preprocessing and Feature Engineering (30 points)

In this section, you will build a reusable preprocessing workflow. This is the main difficulty increase in Lab 2.

## Problem 2.1: Split Before Fitting Transformations (5 points)

**Task:**
1. Use `adult.data` as the development dataset
2. Encode the target variable as binary: `<=50K` -> 0 and `>50K` -> 1
3. Split development data into training and validation subsets before fitting imputers, encoders, or scalers
4. Use stratification so class proportions remain similar
5. Keep `adult.test` as the final test set
6. Report train, validation, and test sizes and class distributions

**Hints:**
- Use `random_state=42`
- A typical validation split is 20-30% of `adult.data`
- Choose KNN settings using validation data, then evaluate once on `adult.test`

## Problem 2.2: Implement `fit_preprocessing` and `transform_preprocessing` (10 points)

**Task:**
1. Implement `fit_preprocessing(train_df, y_train)`
2. Implement `transform_preprocessing(df, preprocessing_state)`
3. Learn missing-value handling from the training split only
4. Learn one-hot encoder state from the training split only
5. Learn target-encoding mappings from the training split only
6. Learn scaler state from the training split only
7. Return numeric feature matrices with identical column order for train, validation, and test

**Required behavior:**
- Low- and moderate-cardinality categorical predictors should use one-hot encoding
- A high-cardinality categorical predictor, such as `native-country`, should use target encoding
- Unknown validation/test categories must not crash transformation
- The final transformed matrices must contain no missing values

**Hints:**
- You may implement target encoding manually with training-split group means, or use the Module 1 `category_encoders.TargetEncoder`
- If using manual target encoding, use the global training positive rate as the fallback for unknown categories
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` is a good starting point for one-hot features
- Do not fit any preprocessing object or mapping on validation or test rows

## Problem 2.3: Target Encoding Comparison for High-Cardinality Categories (8 points)

**Task:**
1. Count unique values in `native-country`
2. Build a comparison of two encoding strategies for `native-country`:
   - one-hot encoding
   - target encoding
3. Report feature count for each strategy
4. Report validation accuracy and F1 for a KNN model using each strategy
5. Track approximate runtime for each strategy
6. Show the training-only target-encoding value for at least five `native-country` categories
7. Explain the fallback value used for unknown validation/test categories

**Hints:**
- Use a fixed simple KNN setting for this comparison, such as `n_neighbors=5`
- This is still a manual validation comparison, not cross-validation
- The goal is to connect dimensionality, leakage risk, runtime, and KNN behavior

## Problem 2.4: Leakage Audit Table (4 points)

**Task:**
Create a leakage audit table and save it under `artifacts/tables/`. The table must include at least these columns:
- `step`
- `fit_on`
- `applied_to`
- `stored_state`
- `leakage_risk`
- `how_risk_was_controlled`

Include rows for:
1. missing-value handling
2. target encoding
3. one-hot encoding
4. scaling
5. KNN model selection
6. final test evaluation

## Problem 2.5: Final Feature Matrix Checks (3 points)

**Task:**
1. Print or save train, validation, and test feature-matrix shapes
2. Confirm all feature matrices have the same columns in the same order
3. Confirm there are no missing or non-numeric values
4. Save a compact feature-matrix summary under `artifacts/tables/`

---

# PART 3: KNN Mechanics From Scratch (20 points)

In this section, you will demonstrate that you understand the KNN algorithm, not just the scikit-learn API.

Use a small subset for this part:
- at most 1,000 training rows
- at most 100 validation rows
- scaled features only
- a small odd `k`, such as 3 or 5

## Problem 3.1: Manual Euclidean Distance Function (5 points)

**Task:**
1. Write a NumPy-based Euclidean distance function
2. Use it to compute distances from one validation row to all rows in the small training subset
3. Identify the nearest 5 training rows
4. Save or print a compact nearest-neighbor example

**Hints:**
- Avoid Python loops over features
- You may use vectorized NumPy operations

## Problem 3.2: Manual KNN Prediction on a Small Subset (8 points)

**Task:**
1. Implement a manual KNN prediction function for binary classification
2. Use majority vote among the `k` nearest neighbors
3. Predict labels for the small validation subset
4. Report accuracy and F1
5. Save a small table with at least 10 manual predictions and their true labels

**Hints:**
- You may loop over validation rows because the subset is intentionally small
- Do not use scikit-learn inside your manual prediction function

## Problem 3.3: Compare Manual KNN to scikit-learn KNN (4 points)

**Task:**
1. Train `KNeighborsClassifier` on the same small training subset
2. Use the same `k` and Euclidean distance
3. Compare manual predictions to scikit-learn predictions
4. Report the number and percentage of matching predictions

## Problem 3.4: Manual KNN Reflection (3 points)

**Task:**
Write 5-8 lines explaining:
1. why manual KNN becomes expensive as data grows
2. which implementation details were easiest to complete
3. which details required you to inspect shapes, labels, or outputs carefully

---

# PART 4: KNN Experiments and Runtime-Aware Model Selection (25 points)

In this section, you will run a more complete scikit-learn KNN experiment while keeping model selection limited to the validation split.

## Problem 4.1: Baseline scikit-learn KNN (5 points)

**Task:**
1. Create a baseline KNN classifier with `n_neighbors=5`
2. Train the model on your final scaled training features
3. Make predictions on the validation set
4. Evaluate using accuracy, precision, recall, F1-score, and `classification_report`
5. Discuss baseline performance compared with the class distribution from Part 1

## Problem 4.2: Distance Diagnostics Before and After Scaling (5 points)

**Task:**
1. Select a fixed sample of validation rows
2. For each selected row, compute nearest-neighbor distance summaries before scaling
3. Repeat after scaling
4. Compare nearest-distance, median-distance, farthest-distance, and nearest/farthest ratio
5. Explain how scaling changes distance behavior

**Hints:**
- Use a small sample to keep runtime controlled
- Features like `fnlwgt` and `capital-gain` can dominate distances without scaling

## Problem 4.3: Encoding Strategy Comparison (5 points)

**Task:**
1. Compare the KNN validation results from the one-hot `native-country` strategy and the target-encoded strategy
2. Include feature count, runtime, accuracy, F1, and a short interpretation
3. Choose the encoding strategy to use for the rest of Lab 2
4. Explain why the chosen strategy is appropriate for KNN

**Hints:**
- Reuse the evidence from Problem 2.3 where appropriate
- Do not choose based on test data

## Problem 4.4: Runtime-Aware `k` and Distance-Metric Validation Loop (7 points)

**Task:**
1. Test at least 10 different `k` values
2. Test at least 2 distance settings, such as:
   - Euclidean: `metric='minkowski', p=2`
   - Manhattan: `metric='minkowski', p=1`
3. For every experiment, record:
   - `k`
   - distance setting
   - validation accuracy
   - validation F1
   - fit time
   - prediction time
4. Create a plot or table showing performance by `k` and distance setting
5. Choose the final KNN configuration using validation evidence
6. Discuss the tradeoff between performance and runtime

**Hints:**
- Use a manual loop, not `GridSearchCV`
- You may use a clearly documented training subset for exploratory runtime comparisons, but final selected-model training must be rerun on the full training split
- Very small `k` can be sensitive to noise
- Very large `k` can smooth over minority-class patterns

## Problem 4.5: Final Test Evaluation (3 points)

**Task:**
1. Freeze your selected preprocessing approach, distance metric, and `k`
2. Train the selected KNN workflow on the full training split
3. Evaluate once on `adult.test`
4. Report accuracy, precision, recall, F1-score, and classification report
5. Compare final test results with validation results and discuss whether they are similar

**Hints:**
- Do not choose a different `k`, distance metric, scaling method, or encoding strategy after seeing test results
- Keep the same feature columns and preprocessing choices used during validation

---

# PART 5: Evaluation, Artifacts, and Reflection (15 points)

In this section, you will summarize the evidence from your KNN workflow.

## Problem 5.1: Confusion Matrices and Metric Interpretation (5 points)

**Task:**
1. Create a confusion matrix for your selected KNN model on the validation set
2. Create a confusion matrix for your selected KNN model on the final test set
3. For the test matrix, report true positives, true negatives, false positives, and false negatives
4. Compare accuracy, precision, recall, and F1-score
5. Explain why accuracy alone would not be enough under class imbalance

## Problem 5.2: Required Artifact Summary Table (5 points)

**Task:**
Create one final summary table and save it under `artifacts/tables/`. It must include:
- baseline KNN validation metrics
- encoding-strategy comparison
- distance-diagnostic summary
- manual KNN vs scikit-learn agreement
- runtime-aware `k` and metric loop winner
- final test metrics
- filenames for related figures or tables

Reference this summary table in `Lab-2-Report.md`.

## Problem 5.3: Final Recommendation and Reflection (5 points)

Complete the final reflection in `Lab-2-Report.md`, not in this handout or the optional notebook.

Address:
1. which KNN configuration you recommend and why
2. whether KNN seems like a strong model for this task
3. how preprocessing choices changed model behavior
4. what leakage risks you controlled
5. one implementation or interpretation detail you had to verify manually
6. two questions you would investigate in later modules

---

## Submission Guidelines

1. Ensure `Lab-2-Assignment.py` runs from top to bottom without errors
2. Save referenced plots, tables, and metrics under `artifacts/`
3. Complete all required sections in `Lab-2-Report.md`
4. Use clear variable names and comments
5. Set `random_state=42` where applicable
6. Create the lab-local `.venv` with `uv venv -p python3.12` and install the included `requirements.txt` with `uv pip install -r requirements.txt`
7. Re-run the script from the activated lab-local `.venv` before submission
8. Make sure your report references the exact artifact filenames created by your script

**Total Time Estimate:** 9-11 hours

**Good luck! This version is still early-publishable after Module 2, but it expects a careful, evidence-backed workflow.**
