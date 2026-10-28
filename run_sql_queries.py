from pathlib import Path
import re
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
SQL_FILE = ROOT / "sql" / "queries.sql"
DB_FILE = ROOT / "data" / "instamart.db"

# Read all seven SQL queries from the saved file.
sql_text = SQL_FILE.read_text(encoding="utf-8")

# Split the file wherever a Q1-Q7 heading appears.
parts = re.split(r"(?m)^-- Q[1-7].*$", sql_text)
queries = [part.strip() for part in parts[1:] if part.strip()]

if len(queries) != 7:
    raise RuntimeError(
        f"Expected 7 SQL queries, but found {len(queries)}."
    )

results = {}

# Execute each query and print its result.
with sqlite3.connect(DB_FILE) as connection:
    for number, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, connection)
        results[number] = result

        print(f"\nQ{number}: {len(result):,} rows")
        print(result.to_string(index=False, max_rows=12))

# ------------------------------------------------------------
# REFERENCE CHECKS
# ------------------------------------------------------------

# Q1: 30 customers.
assert len(results[1]) == 30

# Q2: 30 city/category combinations.
assert len(results[2]) == 30
assert round(results[2]["revenue_inr"].sum(), 2) == 2_998_668.80

largest = results[2].sort_values(
    "revenue_inr",
    ascending=False
).iloc[0]

assert largest["city"] == "Bengaluru"
assert largest["category"] == "Electronics"
assert round(float(largest["revenue_inr"]), 2) == 233_909.45

# Q3: 11 products above the average catalogue price.
assert len(results[3]) == 11

# Q4: 25 customers without completed orders.
assert len(results[4]) == 25

# Q5: 550 customers above the average spend.
assert len(results[5]) == 550
assert round(
    float(results[5]["average_spend_inr"].iloc[0]), 2
) == 2_552.06

# Q6: Check the top-spending customer in each city.
q6 = results[6]

top = (
    q6.sort_values(["city", "spend_rank", "customer_id"])
    .groupby("city", as_index=False)
    .first()
)

expected_top = {
    "Bengaluru": ("C0082", 12856.20),
    "Chennai": ("C0930", 10151.40),
    "Delhi": ("C0291", 13742.60),
    "Hyderabad": ("C0743", 12730.05),
    "Mumbai": ("C0693", 14766.55),
    "Pune": ("C0426", 10377.10),
}

for city, (customer_id, spend) in expected_top.items():
    row = top.loc[top["city"] == city].iloc[0]

    assert row["customer_id"] == customer_id, (
        f"Wrong top customer for {city}"
    )

    assert round(float(row["total_spend_inr"]), 2) == spend, (
        f"Wrong top spend for {city}"
    )

# Exactly eight rows have different RANK and ROW_NUMBER values.
assert int(
    (q6["spend_rank"] != q6["spend_row_number"]).sum()
) == 8

# Q7: 183 days, ending at total completed revenue.
assert len(results[7]) == 183

assert round(
    float(results[7]["cumulative_revenue_inr"].iloc[-1]), 2
) == 2_998_668.80

print("\nALL SEVEN SQL QUERIES AND REFERENCE CHECKS PASSED")