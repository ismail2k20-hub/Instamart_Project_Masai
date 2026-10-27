# Overview

This project builds a customer intelligence and support automation pipeline using deterministic synthetic Instamart data.

The project covers:
- Part A: Data auditing, cleaning, SQLite storage and SQL/pandas analysis.
- Part B: Customer segmentation, lapse prediction and retention recommendations.
- Part C: Support-ticket triage and budget-constrained prioritisation.
- Part D: A simulated refund agent, policy-based decisions, state persistence and crash-and-resume demonstrations.

All source records are synthetic and fictional. All monetary amounts are expressed in INR.

# Reproduce

Run these numbered commands from the repository root in the specified order. Activate the virtual environment before running the project scripts.

1. Create a virtual environment:

   `python -m venv .venv`

2. Activate the virtual environment on Windows PowerShell:

   `.\.venv\Scripts\Activate.ps1`

3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Generate the raw data using the supplied unmodified generator:

   `python generate_data.py`

5. Audit the raw data:

   `python audit_data.py`

6. Clean the data and create the SQLite database:

   `python clean_data.py`

7. Validate the clean data:

   `python validate_data.py`

8. Run the Part A SQL and pandas analysis script:

   `python part_a_analysis.py`

9. Run the Part B pipeline:

   `python part_b.py`

10. Run the Part C pipeline:

    `python part_c.py`

11. Run the Part D agent and recovery demonstrations:

    `python part_d.py`

The script names in steps 8–11 are placeholders until the corresponding scripts are created. Replace them with the actual script names used by the completed project. Every command must work from a clean clone.

# Part A

## Raw data audit

Report raw table shapes, data types, missing-value counts and percentages, and all required audit figures in a table.

## Customer cleaning

Document city and gender standardisation, invalid-age handling, city-median imputation, and signup-date conversion. Report the counts changed by each rule.

## Order cleaning

Document the ordered cleaning rules, including exact duplicate removal, latest-record selection, amount reconstruction, and delivery-time imputation. Report the counts changed by each rule.

## Storage and validation

Report the row counts, order statuses, completed-order revenue and all validation checkpoint results. Document the CSV outputs and SQLite tables.

## SQL analysis

Report the results of Q1–Q7, including the required customer counts, revenue totals, product counts, ranking results and daily running total.

## pandas analysis

Report the pivot-table cross-check, per-city aggregation, descriptive statistics, IQR fences, outlier counts and correlation findings. Keep the four-question exploratory summary within the assignment's 200-word limit.

# Part B

Report the customer-segmentation approach, lapse-model features and evaluation metrics, segment counts, model output paths and retention-list results. Include the required tables and numerical findings from the Part B tasks.

# Part C

Report ticket-triage methodology, classification results, budget assumptions and budget-constrained prioritisation findings. Include the required metrics and the results in `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

# Part D

Report the seven-scenario results in a table, including expected and actual outcomes and refund amounts.

Document the statuses and results of both stop-safely demonstrations.

List the tools executed during the crash-and-resume run, in execution order, and explain how state persistence prevented unsafe duplicate actions.

Include the paths to `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl` and the `state/` folder.

# Limitations and Next Steps

Discuss the limitations of synthetic data, model generalisation, policy interpretation, simulated refund execution, and the absence of real payment-system integration. Describe how future work could improve monitoring, evaluation, security and human oversight.
