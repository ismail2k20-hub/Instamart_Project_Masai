
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 0. PATHS AND REPRODUCIBILITY
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

RANDOM_STATE = 42

customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
products = pd.read_csv(DATA / "products.csv")

required_customer_cols = {
    "customer_id", "city", "age", "gender", "signup_date"
}
required_order_cols = {
    "order_id", "customer_id", "order_ts", "status",
    "order_amount", "discount_pct", "delivery_minutes",
    "product_id",
}
required_product_cols = {"product_id", "category"}

assert required_customer_cols.issubset(customers.columns), (
    f"Missing customer columns: "
    f"{required_customer_cols - set(customers.columns)}"
)
assert required_order_cols.issubset(orders.columns), (
    f"Missing order columns: {required_order_cols - set(orders.columns)}"
)
assert required_product_cols.issubset(products.columns), (
    f"Missing product columns: {required_product_cols - set(products.columns)}"
)

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["status"] = orders["status"].astype(str).str.strip().str.lower()

# Add product category if it is not already present in the clean orders file.
if "category" not in orders.columns:
    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

print("\nPART B INPUT CHECK")
print("------------------")
print("Customers:", len(customers))
print("Clean orders:", len(orders))
print("Products:", len(products))
print("Output folder:", OUTPUT)


# ------------------------------------------------------------
# 1. RFM TABLE: COMPLETED ORDERS ONLY
# ------------------------------------------------------------

completed = orders.loc[orders["status"].eq("completed")].copy()

rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

assert rfm["customer_id"].is_unique
assert rfm[["recency", "frequency", "monetary"]].notna().all().all()

print("\n" + "=" * 65)
print("TASK 1 — RFM SUMMARY")
print("=" * 65)
print("Customers with at least one completed order:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

print("\nReference checks:")
print("Recency median:", round(rfm["recency"].median(), 2), "(reference 19)")
print("Recency minimum:", rfm["recency"].min(), "(reference 0)")
print("Recency maximum:", rfm["recency"].max(), "(reference 180)")
print("Frequency mean:", round(rfm["frequency"].mean(), 2), "(reference 9.46)")
print("Frequency maximum:", rfm["frequency"].max(), "(reference 33)")
print("Monetary mean:", round(rfm["monetary"].mean(), 2), "(reference 2552.06)")
print("Monetary maximum:", round(rfm["monetary"].max(), 2),
      "(reference 14766.55)")


# ------------------------------------------------------------
# 2. STANDARDISE RFM AND EVALUATE K-MEANS K=2,...,8
# ------------------------------------------------------------

rfm_columns = ["recency", "frequency", "monetary"]

rfm_scaler = StandardScaler()
rfm_scaled = rfm_scaler.fit_transform(rfm[rfm_columns])

kmeans_results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=RANDOM_STATE,
    )
    cluster_labels = model.fit_predict(rfm_scaled)

    # Import here so the metric is clearly tied to the cluster labels.
    from sklearn.metrics import silhouette_score

    score = silhouette_score(rfm_scaled, cluster_labels)

    kmeans_results.append({
        "K": k,
        "inertia": model.inertia_,
        "silhouette": score,
    })

kmeans_table = pd.DataFrame(kmeans_results)

print("\n" + "=" * 65)
print("TASK 2 — K-MEANS EVALUATION")
print("=" * 65)
print(kmeans_table.round(3).to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(
    kmeans_table["K"],
    kmeans_table["inertia"],
    marker="o",
)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("K-means Elbow Plot")
plt.xticks(range(2, 9))
plt.tight_layout()
plt.savefig(OUTPUT / "kmeans_elbow.png", dpi=150)
plt.close()

# K=4 is the initial choice because the assignment provides a K=4
# reference profile. We will inspect the printed table and profiles.
CHOSEN_K = 4

final_kmeans = KMeans(
    n_clusters=CHOSEN_K,
    n_init=10,
    random_state=RANDOM_STATE,
)

rfm["segment_id"] = final_kmeans.fit_predict(rfm_scaled)


# ------------------------------------------------------------
# 3. SEGMENT PROFILES, BUSINESS NAMES, PREMIUM FLAG, AND PCA
# ------------------------------------------------------------

profile = (
    rfm.groupby("segment_id")
    .agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .round(2)
)

print("\n" + "=" * 65)
print("TASK 3A — SEGMENT PROFILES")
print("=" * 65)
print(profile.to_string())

# The Premium segment is always the segment with the highest
# mean monetary value. Segment IDs are arbitrary cluster labels,
# not a ranking from worst to best.
premium_id = int(profile["mean_monetary"].idxmax())

# Give the remaining segments distinct names based on their profiles.
remaining_ids = [int(i) for i in profile.index if int(i) != premium_id]

frequent_id = int(
    profile.loc[remaining_ids, "mean_frequency"].idxmax()
)
remaining_ids.remove(frequent_id)

# Of the remaining segments, the one with the greatest recency
# is the most overdue for a completed order.
at_risk_id = int(
    profile.loc[remaining_ids, "mean_recency"].idxmax()
)
remaining_ids.remove(at_risk_id)

name_by_id = {
    premium_id: "Premium",
    frequent_id: "Frequent Regulars",
    at_risk_id: "At-Risk Customers",
}
action_by_id = {
    premium_id: "Offer personalised VIP rewards and protect loyalty.",
    frequent_id: "Use replenishment reminders and relevant repeat-order offers.",
    at_risk_id: "Run a targeted win-back campaign with a clear incentive.",
}

for segment_id in remaining_ids:
    name_by_id[segment_id] = "Occasional Shoppers"
    action_by_id[segment_id] = (
        "Encourage a second or third order with relevant introductory offers."
    )

rfm["segment_name"] = rfm["segment_id"].map(name_by_id)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

profile = profile.reset_index()
profile["segment_name"] = profile["segment_id"].map(name_by_id)
profile["recommended_action"] = profile["segment_id"].map(action_by_id)
profile["is_premium"] = profile["segment_id"].eq(premium_id)

print("\nNamed segment profiles and recommended actions:")
print(profile.to_string(index=False))

print("\nPremium segment ID:", premium_id)
print("Premium segment name:", name_by_id[premium_id])
print("Premium means highest average completed-order monetary value.")
print("Cluster IDs are labels only; their numbers have no natural ordering.")

# PCA is performed on the scaled RFM matrix.
pca = PCA(n_components=2)
pca.fit(rfm_scaled)

pca_ratios = pca.explained_variance_ratio_

print("\nPCA explained variance ratios:")
print("PC1:", round(pca_ratios[0], 4))
print("PC2:", round(pca_ratios[1], 4))
print("Combined:", round(pca_ratios.sum(), 4))

segment_output = rfm[
    [
        "customer_id", "recency", "frequency", "monetary",
        "segment_id", "segment_name", "is_premium",
    ]
].copy()

segment_output.to_csv(OUTPUT / "customer_segments.csv", index=False)

assert list(segment_output.columns) == [
    "customer_id", "recency", "frequency", "monetary",
    "segment_id", "segment_name", "is_premium",
]
assert segment_output["customer_id"].is_unique
assert segment_output["is_premium"].sum() == (
    segment_output["segment_id"].eq(premium_id).sum()
)

print("\nSaved:", OUTPUT / "customer_segments.csv")


# ------------------------------------------------------------
# 4. LAPSE COHORT AND LEAKAGE-SAFE FEATURES
# ------------------------------------------------------------

# All features below are built using orders strictly BEFORE the cutoff.
pre_cutoff = orders.loc[orders["order_ts"] < CUTOFF].copy()

pre_completed = pre_cutoff.loc[
    pre_cutoff["status"].eq("completed")
].copy()

# Cohort: customers with at least one completed order before 1 September.
cohort_ids = pre_completed["customer_id"].drop_duplicates()

# Label: any completed order from 1 September up to, but not including,
# 1 October. These rows are used for the label only, never as features.
label_window = orders.loc[
    orders["order_ts"].ge(CUTOFF)
    & orders["order_ts"].lt(LABEL_END)
    & orders["status"].eq("completed")
].copy()

customers_with_september_order = set(
    label_window["customer_id"].dropna()
)

cohort = customers.loc[
    customers["customer_id"].isin(cohort_ids)
].copy()

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(customers_with_september_order)
).astype(int)

# Historical completed-order features.
historical = (
    pre_completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    .reset_index()
)

historical["recency"] = (
    CUTOFF - historical["last_order_ts"]
).dt.days

# Cancellation rate: cancelled orders divided by all final-status orders
# strictly before the cutoff. Pending orders are excluded.
final_status_orders = pre_cutoff.loc[
    pre_cutoff["status"].isin(["completed", "cancelled"])
].copy()

final_counts = (
    final_status_orders.assign(
        is_cancelled=final_status_orders["status"].eq("cancelled").astype(int)
    )
    .groupby("customer_id")
    .agg(
        final_order_count=("order_id", "nunique"),
        cancelled_order_count=("is_cancelled", "sum"),
    )
    .reset_index()
)

final_counts["cancellation_rate"] = (
    final_counts["cancelled_order_count"]
    / final_counts["final_order_count"].replace(0, np.nan)
)

# Favourite category is computed only from completed orders before cutoff.
category_counts = (
    pre_completed.dropna(subset=["category"])
    .groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_order_count")
    .sort_values(
        ["customer_id", "category_order_count", "category"],
        ascending=[True, False, True],
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# Customer demographics and tenure.
customer_features = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_features["tenure_days"] = (
    CUTOFF - customer_features["signup_date"]
).dt.days

feature_table = (
    cohort[["customer_id", "lapsed"]]
    .merge(historical, on="customer_id", how="left", validate="one_to_one")
    .merge(
        final_counts[["customer_id", "cancellation_rate"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    .merge(category_counts, on="customer_id", how="left", validate="one_to_one")
    .merge(
        customer_features.drop(columns=["signup_date"]),
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
)

feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days",
]

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]
categorical_features = [
    "city",
    "gender",
    "favourite_category",
]

# Use fixed, non-learned fallback values only if the clean data has
# an unexpected missing feature. No cohort-wide imputer is fitted.
for col in numeric_features:
    feature_table[col] = pd.to_numeric(
        feature_table[col], errors="coerce"
    )
    if col in ["avg_discount_pct", "cancellation_rate", "avg_delivery_minutes"]:
        feature_table[col] = feature_table[col].fillna(0)
    elif col == "age":
        feature_table[col] = feature_table[col].fillna(-1)
    else:
        feature_table[col] = feature_table[col].fillna(0)

for col in categorical_features:
    feature_table[col] = (
        feature_table[col].fillna("Unknown").astype(str)
    )
    feature_table[col] = feature_table[col].replace("", "Unknown")

assert feature_table[feature_columns].notna().all().all(), (
    "At least one engineered feature still contains a missing value."
)

X = feature_table[feature_columns].copy()
y = feature_table["lapsed"].astype(int).copy()
customer_ids = feature_table["customer_id"].copy()

print("\n" + "=" * 65)
print("TASK 4 — LAPSE COHORT")
print("=" * 65)
print("Cohort customers:", len(feature_table))
print("Lapsed customers:", int(y.sum()))
print("Lapsed percentage:", round(100 * y.mean(), 2), "%")
print("Feature columns:", feature_columns)
print("Missing feature values:", int(X.isna().sum().sum()))
print(
    "Leakage check: features use only orders dated strictly before",
    CUTOFF.date(),
)
print("Segment labels are NOT included as model features.")

assert X.isna().sum().sum() == 0


# ------------------------------------------------------------
# 5. TRAIN/TEST SPLIT AND TRAINING-ONLY PREPROCESSING
# ------------------------------------------------------------

train_indices, test_indices = train_test_split(
    np.arange(len(feature_table)),
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()
y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

ids_test = customer_ids.iloc[test_indices].reset_index(drop=True)

print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training lapsed:", int(y_train.sum()))
print("Test lapsed:", int(y_test.sum()))

# All scalers and encoders live inside each model's pipeline.
# GridSearchCV fits them on training folds only; after tuning,
# best_estimator_ is refitted on the full training set.
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# Baseline: always predict "not lapsed".
baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\n" + "=" * 65)
print("TASK 5/6 — BASELINE")
print("=" * 65)
print("Baseline accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))
print("Baseline lapsed precision:", round(
    precision_score(y_test, baseline_predictions, zero_division=0), 4
))
print("Baseline lapsed recall:", round(
    recall_score(y_test, baseline_predictions, zero_division=0), 4
))
print("Baseline lapsed F1:", round(
    f1_score(y_test, baseline_predictions, zero_division=0), 4
))
print("Baseline confusion matrix (rows=true [0,1], columns=predicted [0,1]):")
print(confusion_matrix(y_test, baseline_predictions, labels=[0, 1]))


# ------------------------------------------------------------
# 6. GRID SEARCH: LOGISTIC REGRESSION AND RANDOM FOREST
# ------------------------------------------------------------

logistic_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid={
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True,
)

print("\nRunning logistic-regression grid search...")
logistic_grid.fit(X_train, y_train)

print("Logistic best parameters:", logistic_grid.best_params_)
print("Logistic best cross-validated F1:", round(
    logistic_grid.best_score_, 4
))

forest_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    ),
])

forest_grid = GridSearchCV(
    estimator=forest_pipeline,
    param_grid={
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True,
)

print("\nRunning random-forest grid search (9 combinations x 5 folds = 45 fits)...")
forest_grid.fit(X_train, y_train)

print("Random forest best parameters:", forest_grid.best_params_)
print("Random forest best cross-validated F1:", round(
    forest_grid.best_score_, 4
))


# ------------------------------------------------------------
# TEST EVALUATION — AFTER TUNING ONLY
# ------------------------------------------------------------

def report_metrics(label, fitted_model, X_eval, y_eval):
    predictions = fitted_model.predict(X_eval)

    print(f"\n{label}")
    print("Confusion matrix (rows=true [0,1], columns=predicted [0,1]):")
    print(confusion_matrix(y_eval, predictions, labels=[0, 1]))
    print("Accuracy:", round(accuracy_score(y_eval, predictions), 4))
    print("Lapsed precision:", round(
        precision_score(y_eval, predictions, pos_label=1, zero_division=0), 4
    ))
    print("Lapsed recall:", round(
        recall_score(y_eval, predictions, pos_label=1, zero_division=0), 4
    ))
    print("Lapsed F1:", round(
        f1_score(y_eval, predictions, pos_label=1, zero_division=0), 4
    ))

    return predictions


print("\n" + "=" * 65)
print("TASK 6 — UNTOUCHED TEST SET EVALUATION")
print("=" * 65)

lr_test_predictions = report_metrics(
    "Tuned Logistic Regression", logistic_grid.best_estimator_, X_test, y_test
)

rf_test_predictions = report_metrics(
    "Tuned Random Forest", forest_grid.best_estimator_, X_test, y_test
)

lr_test_f1 = f1_score(
    y_test, lr_test_predictions, pos_label=1, zero_division=0
)
rf_test_f1 = f1_score(
    y_test, rf_test_predictions, pos_label=1, zero_division=0
)

# Select the random forest based on its training-only cross-validated F1.
# Its test metrics above are evaluation, not a tuning input.
selected_model_name = "Random Forest"
selected_model = forest_grid.best_estimator_

print("\nSelected model:", selected_model_name)
print("Selected model test F1:", round(rf_test_f1, 4))

if rf_test_f1 < 0.55:
    print(
        "WARNING: Random forest test F1 is below 0.55. "
        "Do not tune against the test set; share the results so we can "
        "check feature construction and the assignment requirements."
    )


# ------------------------------------------------------------
# PRINT TRAINING-ONLY SCALER MEAN FOR FREQUENCY
# ------------------------------------------------------------

fitted_preprocessor = logistic_grid.best_estimator_.named_steps["preprocessor"]
fitted_numeric_scaler = fitted_preprocessor.named_transformers_["numeric"]
frequency_position = numeric_features.index("frequency")

print("\nFrequency scaling check:")
print(
    "Frequency mean learned by logistic model's training-only scaler:",
    round(float(fitted_numeric_scaler.mean_[frequency_position]), 4),
)
print("Full-cohort frequency mean:", round(float(X["frequency"].mean()), 4))
print(
    "These means can differ because the scaler is fitted on training rows "
    "only, whereas the full-cohort mean also includes held-out test rows."
)


# ------------------------------------------------------------
# 7. CLASS-BALANCE EXPERIMENT: OVERSAMPLE TRAINING ROWS ONLY
# ------------------------------------------------------------

# Retain every training row and duplicate only lapsed training rows
# with replacement until the two classes have equal counts.
negative_indices = y_train.index[y_train.eq(0)].to_numpy()
positive_indices = y_train.index[y_train.eq(1)].to_numpy()

if len(positive_indices) < len(negative_indices):
    rng = np.random.RandomState(RANDOM_STATE)
    additional_positive_indices = rng.choice(
        positive_indices,
        size=len(negative_indices) - len(positive_indices),
        replace=True,
    )
    balanced_indices = np.concatenate([
        y_train.index.to_numpy(),
        additional_positive_indices,
    ])
else:
    # Generic fallback in case a different dataset has the opposite balance.
    rng = np.random.RandomState(RANDOM_STATE)
    additional_negative_indices = rng.choice(
        negative_indices,
        size=len(positive_indices) - len(negative_indices),
        replace=True,
    )
    balanced_indices = np.concatenate([
        y_train.index.to_numpy(),
        additional_negative_indices,
    ])

X_train_balanced = X_train.loc[balanced_indices].copy()
y_train_balanced = y_train.loc[balanced_indices].copy()

print("\n" + "=" * 65)
print("TASK 7 — TRAINING-ONLY OVERSAMPLING")
print("=" * 65)
print("Balanced training class counts:")
print(y_train_balanced.value_counts().sort_index().to_dict())

balanced_lr = Pipeline([
    ("preprocessor", make_preprocessor()),
    (
        "classifier",
        LogisticRegression(
            C=logistic_grid.best_params_["classifier__C"],
            max_iter=1000,
        ),
    ),
])

balanced_lr.fit(X_train_balanced, y_train_balanced)

balanced_lr_predictions = balanced_lr.predict(X_test)

print("\nUnbalanced logistic regression on original test set:")
print("Precision:", round(precision_score(
    y_test, lr_test_predictions, zero_division=0
), 4))
print("Recall:", round(recall_score(
    y_test, lr_test_predictions, zero_division=0
), 4))
print("F1:", round(f1_score(
    y_test, lr_test_predictions, zero_division=0
), 4))

print("\nOversampled logistic regression on the same original test set:")
print("Precision:", round(precision_score(
    y_test, balanced_lr_predictions, zero_division=0
), 4))
print("Recall:", round(recall_score(
    y_test, balanced_lr_predictions, zero_division=0
), 4))
print("F1:", round(f1_score(
    y_test, balanced_lr_predictions, zero_division=0
), 4))


# ------------------------------------------------------------
# 8A. RANDOM FOREST IMPORTANCE AND LOGISTIC COEFFICIENTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("TASK 8A — MODEL INTERPRETATION")
print("=" * 65)

rf_preprocessor = forest_grid.best_estimator_.named_steps["preprocessor"]
rf_classifier = forest_grid.best_estimator_.named_steps["classifier"]
rf_feature_names = rf_preprocessor.get_feature_names_out()

importance_table = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_classifier.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop five random-forest features:")
print(importance_table.head(5).round(4).to_string(index=False))

lr_preprocessor = logistic_grid.best_estimator_.named_steps["preprocessor"]
lr_classifier = logistic_grid.best_estimator_.named_steps["classifier"]
lr_feature_names = lr_preprocessor.get_feature_names_out()
lr_coefficients = lr_classifier.coef_[0]

coefficient_table = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_coefficients,
})
coefficient_table["absolute_coefficient"] = (
    coefficient_table["coefficient"].abs()
)
coefficient_table = coefficient_table.sort_values(
    "absolute_coefficient", ascending=False
)

print("\nThree logistic-regression coefficients with largest absolute size:")
print(coefficient_table.head(3)[["feature", "coefficient"]].round(4).to_string(index=False))
print(
    "Interpretation: a positive coefficient raises the model's estimated "
    "lapse log-odds; a negative coefficient lowers them. For one-hot "
    "features, the effect is relative to the dropped reference category."
)


# ------------------------------------------------------------
# 8B. SAVE, RELOAD, AND VERIFY THE SELECTED MODEL
# ------------------------------------------------------------

model_path = OUTPUT / "lapse_model.joblib"
joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

assert np.array_equal(original_predictions, reloaded_predictions), (
    "Reloaded model predictions do not match the original predictions."
)

print("\nModel persistence check:")
print("Saved:", model_path)
print("Reloaded predictions exactly match original predictions: PASS")


# ------------------------------------------------------------
# 8C. PREMIUM TEST CUSTOMERS RANKED FOR RETENTION
# ------------------------------------------------------------

premium_ids = set(
    segment_output.loc[
        segment_output["is_premium"], "customer_id"
    ].astype(str)
)

test_customers = pd.DataFrame({
    "customer_id": ids_test.astype(str).to_numpy(),
    "actually_lapsed": y_test.to_numpy(),
    "lapse_probability": selected_model.predict_proba(X_test)[:, 1],
})

test_customers["is_premium"] = test_customers["customer_id"].isin(premium_ids)

retention_list = (
    test_customers.loc[test_customers["is_premium"]]
    .sort_values("lapse_probability", ascending=False)
    .head(10)
    [["customer_id", "lapse_probability", "actually_lapsed"]]
    .copy()
)

retention_list["lapse_probability"] = retention_list[
    "lapse_probability"
].round(4)

retention_path = OUTPUT / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

print("\nPremium customers on the test set:", int(test_customers["is_premium"].sum()))
print("\nRetention list:")
print(retention_list.to_string(index=False))
print("Truly lapsed customers in this list:", int(retention_list["actually_lapsed"].sum()))
print("Saved:", retention_path)

print(
    "\nLIMITATION: This retention list is illustrative, not a leakage-free "
    "evaluation, because Premium is assigned from whole-window RFM including "
    "September, the same period used to define the lapse label. The model's "
    "predictor features still use only pre-cutoff orders."
)


# ------------------------------------------------------------
# FINAL OUTPUT CHECKS
# ------------------------------------------------------------

assert (OUTPUT / "customer_segments.csv").exists()
assert model_path.exists()
assert retention_path.exists()

print("\n" + "=" * 65)
print("PART B SCRIPT FINISHED")
print("=" * 65)
print("Created outputs/customer_segments.csv")
print("Created outputs/lapse_model.joblib")
print("Created outputs/retention_list.csv")
print("Created outputs/kmeans_elbow.png")
