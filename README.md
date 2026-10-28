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

# Part B

Complete this section after implementing and running the Part B scripts. Report the customer-segmentation approach, lapse-model features and evaluation metrics, segment counts, model output paths and retention-list results. Include the required tables and numerical findings from the Part B tasks.

Required outputs:

- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

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