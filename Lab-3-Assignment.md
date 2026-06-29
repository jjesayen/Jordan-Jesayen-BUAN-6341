# Lab 3 Assignment Handout

> Final graded submissions must use `Lab-3-Assignment.py`, `Lab-3-Report.md`, and referenced files under `artifacts/`. The notebook may be used only as an optional exploration/pre-flow workspace. Do not submit a Jupyter notebook as the final lab submission.

# Lab 3: Leakage-Safe Linear Models for Regression

**Course:** Machine Learning (BUAN 6341)  
**Semester:** Summer 2026  
**Total Points:** 100

---

## Objective

In this lab, you will:
- Build a leakage-safe regression workflow using the Car Prices Poland dataset
- Create train, validation, and final test splits from one raw CSV
- Build reusable preprocessing functions that separate fitting from transformation
- Engineer features with training-only state
- Demonstrate linear-model mechanics on a small subset
- Train baseline and regularized linear models in scikit-learn
- Compare a bounded H2O-3 GLM against the scikit-learn workflow
- Compare model quality using MAE, RMSE, and R-squared
- Interpret coefficients, residuals, and practical model tradeoffs

This lab is scoped to **Module 3, Part 2: Linear and Generalized Linear Models** plus the Module 3 feature-engineering materials needed to make the workflow useful. Later labs cover random forests, broader tuning workflows, advanced model families, and feature-importance methods beyond linear coefficients.

---

## Dataset Information

- **Dataset:** Car Prices Poland
- **File:** `datasets/Car_Prices_Poland.csv`
- **Task:** Regression - predict car `price`

**Primary columns:**
- `mark`: car make
- `model`: car model
- `generation_name`: model generation
- `year`: production year
- `mileage`: mileage
- `vol_engine`: engine volume
- `fuel`: fuel type
- `city`: listing city
- `province`: listing province
- `price`: target variable

**Important Notes:**
- The CSV includes an extra index-like first column. Inspect it and drop it before modeling.
- Create train, validation, and final test splits from the provided CSV.
- Use validation data for model-selection decisions.
- Use the final test split only once after all modeling choices are frozen.
- Use `random_state=42` for reproducibility.
- Do not use `sklearn.pipeline.Pipeline`, `make_pipeline`, `ColumnTransformer`, `GridSearchCV`, or cross-validation helpers in this lab.

---

## Required Design Pattern

Your final code must make the preprocessing workflow inspectable. Implement these functions in `Lab-3-Assignment.py`:

```python
def fit_preprocessing(train_df, y_train):
    """Learn all training-only preprocessing state."""
    return preprocessing_state


def transform_preprocessing(df, preprocessing_state):
    """Apply learned preprocessing state to any compatible split."""
    return X
```

`preprocessing_state` can be a dictionary or another explicit structure. It should contain the values needed to transform validation and test data consistently, such as imputation values, one-hot encoder state, frequency/statistical mappings, rare-category rules, engineered-feature state, feature-column order, and scaler state.

---

## Table of Contents

### PART 1: Data Exploration and Regression Framing (10 points)
- **Problem 1.1:** Load, Clean, and Audit Dataset (4 points)
- **Problem 1.2:** Target Distribution and Feature Review (4 points)
- **Problem 1.3:** Regression Baseline Risk Notes (2 points)

### PART 2: Leakage-Safe Preprocessing and Feature Engineering (30 points)
- **Problem 2.1:** Create Train, Validation, and Final Test Splits (5 points)
- **Problem 2.2:** Implement `fit_preprocessing` and `transform_preprocessing` (10 points)
- **Problem 2.3:** Encoding Strategy for High-Cardinality Categories (8 points)
- **Problem 2.4:** Feature Engineering with Training-Only State (5 points)
- **Problem 2.5:** Leakage Audit and Feature Matrix Checks (2 points)

### PART 3: Linear Regression Mechanics From Scratch (20 points)
- **Problem 3.1:** Manual Prediction Equation on a Small Subset (5 points)
- **Problem 3.2:** Manual Regression Metrics (6 points)
- **Problem 3.3:** Compare Manual Calculations to scikit-learn (5 points)
- **Problem 3.4:** Mechanics Reflection (4 points)

### PART 4: Linear-Model Experiments and Runtime-Aware Model Selection (25 points)
- **Problem 4.1:** Baseline scikit-learn Linear Regression (5 points)
- **Problem 4.2:** Ridge and Lasso Validation Loops (8 points)
- **Problem 4.3:** Residual Diagnostics and Coefficient Review (4 points)
- **Problem 4.4:** Bounded H2O-3 GLM Comparison (5 points)
- **Problem 4.5:** Final Test Evaluation (3 points)

### PART 5: Comparison, Artifacts, and Reflection (15 points)
- **Problem 5.1:** Cross-Framework Model Comparison (5 points)
- **Problem 5.2:** Coefficient and Residual Interpretation (5 points)
- **Problem 5.3:** Required Artifact Summary and Reflection (5 points)

**Total Points: 100**

---

## Local Runtime

Before running your script, create a lab-local environment and install this lab's requirements:

```bash
uv venv -p python3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python Lab-3-Assignment.py
```

Optional stricter run command:

```bash
uv run --active python Lab-3-Assignment.py
```

---

# PART 1: Data Exploration and Regression Framing (10 points)

In this section, you will load the dataset, identify modeling risks, and establish simple baseline performance before fitting machine-learning models.

## Problem 1.1: Load, Clean, and Audit Dataset (4 points)

**Task:**
1. Load `datasets/Car_Prices_Poland.csv` into a pandas DataFrame.
2. Display shape, columns, data types, and the first 5 rows.
3. Identify the extra index-like first column and drop it.
4. Confirm the cleaned dataset contains the expected predictor columns and target column.
5. Report final shape after cleanup.

**Hints:**
- Start with `pd.read_csv()`.
- The extra index-like column may appear as `Unnamed: 0` or as an empty first column depending on how pandas reads the CSV.

## Problem 1.2: Target Distribution and Feature Review (4 points)

**Task:**
1. Confirm `price` is numeric.
2. Plot and summarize the `price` distribution.
3. Separate numeric and categorical predictors.
4. Report cardinality for categorical features.
5. Identify high-cardinality columns such as `model` or `city` and discuss implications for linear models.
6. Save at least one target-distribution or data-quality artifact under `artifacts/`.

**Discussion:**

Write 4-6 lines in `Lab-3-Report.md` on target distribution shape, outliers, and cardinality challenges.

## Problem 1.3: Regression Baseline Risk Notes (2 points)

**Task:**
1. Identify the simple mean-target baseline you will use after splitting.
2. Explain why a baseline is necessary before model training.

**Report Requirement:**

Write 5-8 lines in `Lab-3-Report.md` explaining which data properties may make linear regression difficult here. Address at least target skew, high-cardinality categories, outliers, and possible non-linear relationships.

---

# PART 2: Leakage-Safe Preprocessing and Feature Engineering (30 points)

In this section, you will build a reusable preprocessing workflow. This is the main difficulty increase in Lab 3.

## Problem 2.1: Create Train, Validation, and Final Test Splits (5 points)

**Task:**
1. Separate features `X` and target `y`.
2. Create a final test split first, such as 20% of the cleaned dataset.
3. Split the remaining development data into training and validation subsets.
4. Use `random_state=42`.
5. Report train, validation, and test sizes.
6. Do not fit any preprocessing object or mapping before the split.

**Hints:**
- A reasonable split is 64% train, 16% validation, and 20% final test.
- Choose Ridge and Lasso settings on validation data, then compare a bounded H2O baseline and regularized GLM.
- Use the final test split only after model choices are frozen.

## Problem 2.2: Implement `fit_preprocessing` and `transform_preprocessing` (10 points)

**Task:**
1. Implement `fit_preprocessing(train_df, y_train)`.
2. Implement `transform_preprocessing(df, preprocessing_state)`.
3. Learn missing-value handling from the training split only.
4. Learn one-hot encoder state from the training split only.
5. Learn high-cardinality frequency or statistical mappings from the training split only.
6. Learn engineered-feature state from the training split only.
7. Learn scaler state from the training split only.
8. Return numeric feature matrices with identical column order for train, validation, and test.

**Required behavior:**
- Low-cardinality categorical predictors should use `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- At least one high-cardinality predictor, such as `model` or `city`, must use a deliberate strategy such as frequency encoding, rare-category grouping, or a training-only statistical feature.
- Unknown validation/test categories must not crash transformation.
- The final transformed matrices must contain no missing values.
- Do not use `pd.get_dummies()`, pipeline abstractions, or column-transformer abstractions.

## Problem 2.3: Encoding Strategy for High-Cardinality Categories (8 points)

**Task:**
1. Count unique values in `model`, `city`, and any other high-cardinality categorical columns.
2. Compare at least two high-cardinality strategies for one chosen column:
   - frequency encoding
   - rare-category grouping plus one-hot encoding
   - training-only target/statistical encoding, if you choose to attempt it
3. Report feature count for each strategy.
4. Report validation MAE and RMSE for a simple linear model under each strategy.
5. Track approximate runtime for each strategy.
6. Explain the fallback value used for unknown validation/test categories.

**Hints:**
- If you use target/statistical encoding, compute mappings from training rows only.
- For linear models, target statistics can overfit when groups are small; explain how you controlled that risk.

## Problem 2.4: Feature Engineering with Training-Only State (5 points)

**Task:**
1. Create at least 3 engineered features using Module 3 feature-engineering ideas.
2. Include at least one transformation, such as `log_mileage` or `car_age`.
3. Include at least one interaction, such as `engine_liters * car_age`.
4. Include at least one aggregation or statistical feature computed from training rows only, such as mean training price by `mark` with a global fallback.
5. Apply all engineered features to train, validation, and test consistently.
6. Briefly assess whether engineered features improved validation performance.

## Problem 2.5: Leakage Audit and Feature Matrix Checks (2 points)

**Task:**
1. Create a leakage audit table and save it under `artifacts/tables/`.
2. Print or save train, validation, and final test feature-matrix shapes.
3. Verify all three transformed matrices have identical columns in identical order.
4. Verify there are no missing values after transformation.

Your leakage audit table must include at least these columns:
- `step`
- `fit_on`
- `applied_to`
- `stored_state`
- `leakage_risk`
- `how_risk_was_controlled`

Include rows for:
1. train/validation/test splitting
2. missing-value handling
3. one-hot encoding
4. high-cardinality encoding
5. engineered statistical features
6. scaling
7. model-selection loops
8. final test evaluation

---

# PART 3: Linear Regression Mechanics From Scratch (20 points)

In this section, you will demonstrate the mechanics of a linear regression model on a small, inspectable subset before relying on full library workflows. Keep this section small enough to verify manually.

## Problem 3.1: Manual Prediction Equation on a Small Subset (5 points)

**Task:**
1. Select 2-3 numeric predictors from your transformed training matrix.
2. Fit a small `LinearRegression` model on a small training subset.
3. Extract the intercept and coefficients.
4. Manually compute predictions for at least 10 validation rows using the equation `prediction = intercept + coefficient_1*x_1 + ...`.
5. Save a table showing manual predictions, scikit-learn predictions, actual prices, and residuals.

## Problem 3.2: Manual Regression Metrics (6 points)

**Task:**
1. Implement small manual functions for MAE and RMSE.
2. Compute MAE and RMSE for the manual-prediction table from Problem 3.1.
3. Compute residuals for the same rows.
4. Compare your manual metric results to scikit-learn metric functions.
5. Explain any rounding differences.

## Problem 3.3: Compare Manual Calculations to scikit-learn (5 points)

**Task:**
1. Report the maximum absolute difference between manual predictions and scikit-learn predictions on the subset.
2. Confirm that your manual predictions match scikit-learn within a small tolerance.
3. Save the comparison table under `artifacts/tables/`.
4. Explain how intercept, coefficient, feature scale, and residual relate to each other.

## Problem 3.4: Mechanics Reflection (4 points)

**Task:**
Write 6-8 lines in `Lab-3-Report.md` explaining what the manual mechanics exercise showed that would be hidden if you only called `.fit()` and `.predict()`.

---

# PART 4: Linear-Model Experiments and Runtime-Aware Model Selection (25 points)

In this section, you will train full linear-model workflows, use validation results to select settings, and evaluate the selected models once on the final test split.

## Problem 4.1: Baseline scikit-learn Linear Regression (5 points)

**Task:**
1. Use the full manually prepared feature matrices from Part 2.
2. Train `LinearRegression` on the training data.
3. Predict on train and validation data.
4. Compute MAE, RMSE, and R-squared for both splits.
5. Save actual-vs-predicted and residual plots for the validation set.

## Problem 4.2: Ridge and Lasso Validation Loops (8 points)

**Task:**
1. Train `Ridge` models using an explicit validation loop over candidate alpha values.
2. Train `Lasso` models using an explicit validation loop over candidate alpha values.
3. Report train and validation MAE, RMSE, and R-squared for each alpha.
4. Report the number of zero coefficients for each Lasso alpha.
5. Save one combined validation table under `artifacts/tables/`.
6. Select one final scikit-learn model based on validation evidence.

**Hint:**
- Try alphas like `[0.001, 0.01, 0.1, 1, 10, 100]`.

## Problem 4.3: Residual Diagnostics and Coefficient Review (4 points)

**Task:**
1. Create residual and actual-vs-predicted plots for the selected scikit-learn model.
2. Identify the largest positive and negative residuals.
3. Identify the strongest positive and negative coefficient patterns.
4. Explain one limitation of coefficient interpretation in this encoded feature space.

## Problem 4.4: Bounded H2O-3 GLM Comparison (5 points)

**Task:**
1. Start a bounded local H2O cluster.
2. Convert cleaned pandas train, validation, and final test splits to H2OFrames.
3. Cast categorical predictors to factors using `.asfactor()`.
4. Train one baseline H2O GLM and one regularized H2O GLM.
5. Compare their validation MAE, RMSE, and R-squared.
6. Shut down the H2O cluster at the end of script execution.

**Required setup:**

```python
h2o.init(name="ML_Project_Cluster", max_mem_size="4G", nthreads=4, verbose=False)
```

Manual standardization/scaling is not required for H2O-3 GLM before training.

## Problem 4.5: Final Test Evaluation (3 points)

**Task:**
1. Evaluate the selected final scikit-learn model once on the final test split.
2. Evaluate the selected H2O model once on the final test split.
3. Report final test MAE, RMSE, and R-squared for both.
4. Do not revise model choices after seeing final test results.

---

# PART 5: Comparison, Artifacts, and Reflection (15 points)

## Problem 5.1: Cross-Framework Model Comparison (5 points)

**Task:**
1. Create one summary table containing final scikit-learn and H2O model metrics.
2. Include at least MAE, RMSE, and R-squared on validation and final test data.
3. Identify which framework/model performed best and by which metric.
4. Save the comparison table under `artifacts/tables/`.

## Problem 5.2: Coefficient and Residual Interpretation (5 points)

**Task:**
1. Interpret top influential features from your final models.
2. Analyze residual behavior, including bias pattern, heteroscedasticity, and outliers.
3. Discuss one business-facing takeaway for pricing strategy.
4. Propose one next modeling step beyond linear models.

## Problem 5.3: Required Artifact Summary and Reflection (5 points)

**Task:**
1. Create an artifact summary table listing every plot, table, or output file referenced in `Lab-3-Report.md`.
2. Save the artifact summary under `artifacts/tables/`.
3. Write a final reflection in `Lab-3-Report.md`.

**Final Reflection Requirement:**

Write 8-12 lines covering model quality, leakage controls, limitations, verification steps, and next steps.

---

# Final Submission Checklist

Before submitting, make sure you have:

- [ ] Completed final code in `Lab-3-Assignment.py`
- [ ] Created `Lab-3-Report.md` from `Lab-3-Report-Template.md`
- [ ] Filled in your name and NetID in `Lab-3-Report.md`
- [ ] Saved referenced plots, tables, metrics, or other outputs under `artifacts/`
- [ ] Run `python Lab-3-Assignment.py` from the lab-local `.venv`
- [ ] Verified the final files on GitHub after pushing
- [ ] Left `Lab-3-Assignment.ipynb` out of the final submission unless your instructor explicitly asked for it
