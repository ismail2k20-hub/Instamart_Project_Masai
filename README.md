# Overview

This project builds a customer intelligence and support automation pipeline using deterministic synthetic Instamart data.

The project covers:
- Part A: Data auditing, cleaning, SQLite storage, SQL analysis and pandas exploratory analysis.
- Part B: Customer segmentation, lapse prediction and retention recommendations.
- Part C: Support-ticket triage and budget-constrained prioritisation.
- Part D: A simulated refund agent, policy-based decisions, state persistence and crash-and-resume demonstrations.

All source records are synthetic and fictional. All monetary amounts are expressed in INR.

# Reproduce

Run these commands from the repository root in Windows PowerShell. Run each command in order. Do not include the PowerShell prompt itself when copying commands.

1. Create a Python virtual environment:

   `python -m venv .venv`

2. Activate the virtual environment:

   `.\.venv\Scripts\Activate.ps1`

3. Install the required libraries:

   `python -m pip install -r requirements.txt`

4. Generate the raw data using the supplied, unmodified generator:

   `python generate_data.py`

   Expected final line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`

5. Audit the raw data:

   `python audit_data.py`

6. Clean the data, save the clean CSV files and create the SQLite database:

   `python clean_data.py`

7. Validate the clean data and database:

   `python validate_data.py`

   Expected success message: `ALL CHECKPOINTS PASSED`

8. Run the seven SQL queries:

   `python run_sql_queries.py`

   Expected success message: `ALL SEVEN SQL QUERIES AND REFERENCE CHECKS PASSED`

9. Run the pandas analysis and cross-checks:

   `python part_a_analysis.py`

   Expected final success message: `PART A SQL AND PANDAS CHECKS PASSED`

These commands reproduce Part A. Add the Part B, Part C and Part D commands after their scripts have been implemented and tested. A clean clone must include all required scripts, dependencies and source files.

# Part A

## Raw data audit

The generator creates 1,200 customers, 40 products, 13,084 raw order rows and 164 support tickets.

| Raw audit check | Result |
|---|---:|
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric or missing order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Missing customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Blank customer genders | 30 |
| Empty ticket texts | 4 |
| Ticket texts over 500 words | 4 |

The audit script also prints each raw table's shape, data types, missing-value counts and missing-value percentages. These are reported directly by `audit_data.py`.

## Customer cleaning

- City names were stripped of surrounding spaces and standardized to title case. **476** city values changed.
- Gender values were standardized to `female`, `male` or `unknown`. **950** values changed under the script's comparison rule.
- Ages below 10 or above 100 were treated as invalid. **6** invalid ages were set to missing.
- Missing and invalid ages were filled using the median age of the corresponding cleaned city. **67** ages were imputed.
- Signup dates were parsed and formatted as `YYYY-MM-DD`. **0** values changed under the script's comparison rule.
- All 1,200 customer records were retained.

## Order cleaning

The following rules were applied in order:

1. Removed **119** exact duplicate order rows.
2. Parsed order timestamps and sorted records by order ID and timestamp.
3. Kept the latest timestamp for each order ID. After exact duplicate removal, 1,312 rows belonged to repeated order IDs; retaining one row per ID removed 656 additional rows.
4. Converted quantity, discount percentage, order amount and delivery time to numeric values where possible.
5. Reconstructed **112** missing or unparseable order amounts using catalogue price × quantity × (1 − discount percentage / 100), rounded to two decimal places.
6. Set **105** negative delivery times to missing after selecting the latest order records.
7. Cleared delivery times for cancelled orders. No non-missing cancelled delivery times required clearing in this dataset.
8. Imputed **774** missing delivery times for completed orders using the median valid delivery time for the relevant product category.
9. Stored order timestamps as text in `YYYY-MM-DD HH:MM:SS` format.

After cleaning, 12,309 unique order IDs remain. The final order statuses are 11,110 completed and 1,199 cancelled orders. Pending orders are removed because only the latest record for each order ID is retained.

## Storage and validation

The following outputs are created:

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`

The SQLite database contains three tables: `customers`, `products` and `orders`.

| Validation checkpoint | Expected result |
|---|---:|
| Clean customer rows | 1,200 |
| Clean product rows | 40 |
| Clean order rows | 12,309 |
| Unique customer IDs | 1,200 |
| Unique product IDs | 40 |
| Unique order IDs | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |
| Missing order amounts | 0 |
| Completed orders with missing delivery time | 0 |
| Cancelled orders with a delivery time | 0 |

The validator checks customer city counts, gender counts, valid ages, product counts, order statuses, revenue, timestamp format and SQLite table counts. It prints `ALL CHECKPOINTS PASSED` only when every assertion succeeds.

## SQL analysis

The seven queries are saved in `sql/queries.sql`. The script `run_sql_queries.py` runs each query against `data/instamart.db` using `pandas.read_sql_query`.

| Query | Result |
|---|---|
| Q1: Completed orders above INR 500, grouped by customer with qualifying totals above INR 8,000 | 30 customers |
| Q2: Completed revenue by city and category | 30 city/category combinations; total INR 2,998,668.80 |
| Q2: Largest city/category revenue combination | Bengaluru / Electronics, INR 233,909.45 |
| Q3: Products priced above average catalogue price | 11 products |
| Q4: Customers without a completed order | 25 customers |
| Q5: Customers spending above the average completed spend of customers with completed orders | 550 customers; average INR 2,552.06 |
| Q6: City-based customer spend rankings using `RANK`, `DENSE_RANK` and `ROW_NUMBER` | 8 rows have different `RANK` and `ROW_NUMBER` values |
| Q7: Daily completed revenue with cumulative revenue | 183 days; final cumulative revenue INR 2,998,668.80 |

The highest-spending customer in each city is:

| City | Customer ID | Completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

`RANK()` assigns the same rank to tied values and leaves gaps after ties, whereas `ROW_NUMBER()` assigns a unique row number to every row. Therefore, the functions can return different values when customers tie on spend.

## pandas analysis

The pandas analysis independently cross-checks Q2 using `pd.pivot_table` with `margins=True`. The pivot-table grand total matches SQL revenue at **INR 2,998,668.80**.

| City | Completed revenue | Mean order amount | Completed order count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |

For completed orders:

| Statistic | Result |
|---|---:|
| Mean order amount | INR 269.91 |
| Median order amount | INR 135.00 |
| Skewness | 4.61 |
| First quartile (Q1) | INR 64.80 |
| Third quartile (Q3) | INR 297.00 |
| Upper 1.5 × IQR fence | INR 645.30 |
| Outliers below the lower fence | 0 |
| Outliers above the upper fence | 897 |

The completed-order correlation matrix covers order amount, discount percentage, quantity and delivery time. The largest absolute correlation is between order amount and discount percentage, approximately -0.095. This is a weak correlation and does not establish causation.

### Four-question exploratory summary

Completed-order amounts are strongly right-skewed: the mean of INR 269.91 is substantially higher than the median of INR 135.00, and skewness is approximately 4.61. The upper 1.5 × IQR fence is INR 645.30, with 897 observations above it and none below the lower fence. Bengaluru has completed revenue of INR 523,643.40 across 1,894 orders, while Mumbai has INR 479,058.50 across 1,613 orders. The city/category pivot table agrees with the SQL revenue total of INR 2,998,668.80. The largest absolute correlation is between order amount and discount percentage at approximately -0.095; this relationship is weak, and correlation does not demonstrate causation.

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Table

The RFM table was constructed from completed orders only, using the snapshot moment `2025-10-01 00:00:00`.

- **Recency:** Whole days between the snapshot and the customer's most recent completed order.
- **Frequency:** Number of distinct completed orders.
- **Monetary:** Sum of completed-order amounts in INR.

The RFM table contains 1,175 customers with at least one completed order.

| Statistic | Recency (days) | Frequency | Monetary (INR) |
|---|---:|---:|---:|
| Mean | 33.56 | 9.46 | 2,552.06 |
| Median | 19 | 5 | 2,286.50 |
| Minimum | 0 | 1 | 12.00 |
| Maximum | 180 | 33 | 14,766.55 |

### 2. K-Means Clustering

The three RFM features were standardised using `StandardScaler`. K-means was evaluated for K=2 through K=8 using `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.761 | 0.465 |
| 3 | 1268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

K=4 was selected because it has the highest silhouette score among the evaluated values and provides four actionable customer groups. The elbow plot is saved as `outputs/kmeans_elbow.png`.

Inertia decreases as K increases because more clusters allow customers to be closer to their assigned centroids. Therefore, K=8 having lower inertia than K=2 does not automatically make it the best choice; silhouette scores, the elbow and business interpretability must also be considered.

### 3. Segment Profiles and Business Actions

The final K=4 solution produced the following segment profiles.

| Segment ID | Segment name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Encourage repeat orders with relevant introductory offers. |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | Offer personalised VIP rewards and protect loyalty. |
| 2 | Frequent Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Use replenishment reminders and relevant repeat-order offers. |
| 3 | At-Risk Customers | 182 | 109.04 | 2.20 | 685.99 | Run a targeted win-back campaign with a clear incentive. |

The Premium segment is the segment with the highest mean monetary value. Cluster IDs are arbitrary labels and do not represent an ordering of customer value.

PCA was applied to the scaled RFM matrix.

- PC1 explained variance ratio: 0.6343
- PC2 explained variance ratio: 0.2180
- Combined explained variance: 0.8523

The segment assignments and RFM values were saved to `outputs/customer_segments.csv` with the required columns: `customer_id`, `recency`, `frequency`, `monetary`, `segment_id`, `segment_name` and `is_premium`.

### 4. Lapse Definition and Feature Engineering

The lapse cutoff is `2025-09-01`. The cohort contains customers with at least one completed order strictly before the cutoff. A customer is labelled `lapsed = 1` if they have no completed order from 1 September through 30 September 2025; otherwise, `lapsed = 0`.

- Cohort size: 1,156 customers
- Lapsed customers: 443 (38.32%)
- Training set: 924 rows
- Test set: 232 rows

The features include recency, frequency, monetary, average discount percentage, cancellation rate, average delivery time, favourite product category, city, age, gender and tenure in days.

No orders dated on or after `2025-09-01` were used to construct model features. September completed orders were used only to define the target label. Segment labels from Task 3 were not used as model features.

### 5. Preprocessing and Baseline

The data was split using `train_test_split(test_size=0.2, stratify=y, random_state=42)`.

Numeric features were standardised and nominal features (`city`, `gender` and favourite product category) were one-hot encoded using `OneHotEncoder(drop="first", handle_unknown="ignore")`. Preprocessing was placed inside scikit-learn pipelines and fitted on training rows only. Grid-search cross-validation also fitted each pipeline within its training folds; the held-out test rows were not used to fit encoders or scalers.

The feature table was checked to confirm that no feature column contained a missing value.

The frequency mean learned by the logistic-regression training-only scaler was 8.1147, compared with the full-cohort mean of 8.0285. These differ because the scaler is fitted on training rows only, whereas the full-cohort mean includes the held-out test rows.

The baseline always predicted `not lapsed`:

- Accuracy: 0.6164
- Lapsed-class precision: 0.0000
- Lapsed-class recall: 0.0000
- Lapsed-class F1: 0.0000

### 6. Model Tuning and Test Evaluation

Both models were tuned using `GridSearchCV`, five-fold cross-validation and `scoring="f1"`.

**Logistic regression**

- Model: `LogisticRegression(max_iter=1000)`
- C grid: `{0.01, 0.1, 1, 10, 100}`
- Best C: 100
- Best cross-validated F1: 0.6506

**Random forest**

- Model: `RandomForestClassifier(n_estimators=100, random_state=42)`
- Maximum depth grid: `{3, 5, 8}`
- Minimum leaf-size grid: `{5, 10, 20}`
- Total search: 9 parameter combinations and 45 fits
- Best parameters: `max_depth=8`, `min_samples_leaf=10`
- Best cross-validated F1: 0.6811

The following metrics were calculated on the untouched test set after tuning. Confusion matrices use rows for true classes `[0, 1]` and columns for predicted classes `[0, 1]`.

**Logistic regression confusion matrix**

|  | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 111 | 32 |
| Actual lapsed | 31 | 58 |

- Accuracy: 0.7284
- Lapsed-class precision: 0.6444
- Lapsed-class recall: 0.6517
- Lapsed-class F1: 0.6480

**Random forest confusion matrix**

|  | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 113 | 30 |
| Actual lapsed | 31 | 58 |

- Accuracy: 0.7371
- Lapsed-class precision: 0.6591
- Lapsed-class recall: 0.6517
- Lapsed-class F1: 0.6554

**Selected model: Random forest.** It achieved a cross-validated F1 of 0.6811 and a test F1 of 0.6554, exceeding the required minimum of 0.55. Missing a customer who is about to lapse can mean losing future orders and revenue, while wrongly targeting a loyal customer can waste campaign budget or provide an unnecessary discount. The random forest had slightly higher test precision and F1 than logistic regression, with the same recall, so it is the selected model for this implementation. Its results should be monitored on future customer data before using it for operational decisions.

### 7. Class-Balance Experiment

The training set was oversampled by retaining all original training rows and duplicating lapsed training rows with replacement until the classes were balanced. Oversampling used `random_state=42`; the original test set remained unchanged.

The balanced training set contained 570 not-lapsed and 570 lapsed rows.

| Logistic-regression model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original, unbalanced training | 0.6444 | 0.6517 | 0.6480 |
| Oversampled training | 0.5891 | 0.8539 | 0.6972 |

Recall moved the most, increasing from 0.6517 to 0.8539, while precision decreased from 0.6444 to 0.5891. This means the retention team could identify more customers who lapse, but would also target more customers who would not lapse. Campaign costs and the value of successfully retained customers should be considered when choosing a targeting threshold.

### 8. Feature Interpretation, Model Persistence and Retention List

The five most important random-forest features were:

| Feature | Importance |
|---|---:|
| Frequency | 0.2796 |
| Monetary | 0.2712 |
| Recency | 0.1152 |
| Average delivery minutes | 0.0612 |
| Cancellation rate | 0.0525 |

Frequency and monetary value were the strongest features in this fitted model. Recency was the third most important feature, while average delivery time and cancellation rate also contributed to the predictions. Feature importance indicates the model's relative use of a feature; it does not establish causation.

The three logistic-regression coefficients with the largest absolute magnitudes were:

| Feature | Coefficient | Interpretation |
|---|---:|---|
| Frequency | -1.6973 | Higher frequency is associated with lower predicted lapse log-odds, holding the other encoded features constant. |
| Favourite category: Electronics | -1.1351 | This category has lower predicted lapse log-odds than the dropped reference category, conditional on the other features. |
| Favourite category: Grocery | -0.4492 | This category has lower predicted lapse log-odds than the dropped reference category, conditional on the other features. |

The selected random-forest pipeline was saved to `outputs/lapse_model.joblib`, reloaded using `joblib.load`, and checked to confirm that its test predictions exactly matched those of the original fitted model.

The top 10 Premium customers in the test set were ranked by predicted lapse probability and saved to `outputs/retention_list.csv`. There were 40 Premium customers in the test set, and 1 of the top 10 ranked customers actually lapsed.

The retention list is illustrative and not a leakage-free evaluation because the Premium flag comes from whole-window RFM, including September orders that also define the lapse label. The model's predictor features themselves use only orders dated before `2025-09-01`.


# Part C

Complete this section after implementing and running the Part C scripts. Report the ticket-triage methodology, classification results, budget assumptions and budget-constrained prioritisation findings. Include the required metrics and numerical results.

Required outputs:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

# Part D

Complete this section after implementing and running the Part D scripts.

Include:
- A table showing all seven scenarios, expected outcomes, actual outcomes, statuses and refund amounts.
- The status and result of each of the two stop-safely demonstrations.
- The tools executed during the crash-and-resume run, in execution order.
- An explanation of how persistent state prevented unsafe duplicate actions.

Required outputs:

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/`

# Limitations and Next Steps

The project uses deterministic synthetic data, so its results may not represent real customer behaviour or real operational conditions. Model performance on synthetic data does not guarantee generalisation to real customers. Policy-based decisions may require human review when rules are ambiguous or cases fall outside known scenarios. The refund agent is simulated and does not integrate with a real payment system.

Future work should evaluate models on representative real-world data where legally permitted, monitor data quality and model drift, strengthen logging and access controls, test recovery behaviour, and include human oversight for high-impact or ambiguous decisions.

The Premium flag is a post-hoc whole-window label based on RFM that includes September orders, which also define the lapse label. Therefore, the retention list is illustrative and not a leakage-free evaluation. The lapse model's features use only orders dated before 2025-09-01, and the segment labels are not used as model features.