
"""Validate raw-data audit figures and cleaned-data checkpoints."""

import sqlite3

import pandas as pd

RAW = "data/raw"
CLEAN = "data/clean"
DB = "data/instamart.db"

# ----------------------------------------------------------
# Check raw audit figures
# ----------------------------------------------------------

raw_orders = pd.read_csv(f"{RAW}/orders.csv")
raw_customers = pd.read_csv(f"{RAW}/customers.csv")

raw_amounts = pd.to_numeric(
    raw_orders["order_amount"],
    errors="coerce"
)

raw_delivery = pd.to_numeric(
    raw_orders["delivery_minutes"],
    errors="coerce"
)

raw_age = pd.to_numeric(
    raw_customers["age"].replace("", pd.NA),
    errors="coerce"
)

raw_gender = (
    raw_customers["gender"]
    .fillna("")
    .astype(str)
    .str.strip()
)

raw_city = raw_customers["city"]

checks = {}

checks["raw_order_rows"] = len(raw_orders) == 13084
checks["exact_duplicate_rows"] = (
    int(raw_orders.duplicated().sum()) == 119
)
checks["pending_rows"] = (
    int(raw_orders["status"].eq("pending").sum()) == 656
)
checks["distinct_order_ids"] = (
    int(raw_orders["order_id"].nunique()) == 12309
)
checks["non_numeric_amounts"] = (
    int(raw_amounts.isna().sum()) == 120
)
checks["negative_delivery_times"] = (
    int(raw_delivery.lt(0).sum()) == 106
)
checks["blank_delivery_times"] = (
    int(
        raw_orders["delivery_minutes"].isna().sum()
        + raw_orders["delivery_minutes"].eq("").sum()
    ) == 2541
)
checks["blank_ages"] = (
    int(
        raw_customers["age"].isna().sum()
        + raw_customers["age"].eq("").sum()
    ) == 61
)
checks["invalid_ages"] = (
    int(
        raw_age.lt(10).fillna(False).sum()
        + raw_age.gt(100).fillna(False).sum()
    ) == 6
)
checks["raw_city_spellings"] = (
    int(raw_city.nunique()) == 24
)
checks["raw_gender_spellings"] = (
    int(raw_gender[raw_gender.ne("")].nunique()) == 10
)
checks["blank_genders"] = (
    int(raw_gender.eq("").sum()) == 30
)

# ----------------------------------------------------------
# Check cleaned CSV files
# ----------------------------------------------------------

customers = pd.read_csv(f"{CLEAN}/customers.csv")
products = pd.read_csv(f"{CLEAN}/products.csv")
orders = pd.read_csv(f"{CLEAN}/orders.csv")

checks["clean_customer_rows"] = len(customers) == 1200
checks["clean_product_rows"] = len(products) == 40
checks["clean_order_rows"] = len(orders) == 12309
checks["unique_order_ids"] = orders["order_id"].is_unique
checks["six_cities"] = customers["city"].nunique() == 6

checks["city_counts"] = (
    customers["city"].value_counts().to_dict()
    == {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
)

checks["gender_counts"] = (
    customers["gender"].value_counts().to_dict()
    == {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }
)

checks["no_missing_ages"] = customers["age"].notna().all()
checks["ages_between_18_and_60"] = (
    pd.to_numeric(customers["age"]).between(18, 60).all()
)

checks["signup_dates_parse"] = (
    pd.to_datetime(
        customers["signup_date"],
        errors="coerce"
    ).notna().all()
)

checks["no_pending_orders"] = (
    not orders["status"].eq("pending").any()
)

checks["completed_count"] = (
    int(orders["status"].eq("completed").sum()) == 11110
)

checks["cancelled_count"] = (
    int(orders["status"].eq("cancelled").sum()) == 1199
)

checks["no_missing_order_amounts"] = (
    pd.to_numeric(
        orders["order_amount"],
        errors="coerce"
    ).notna().all()
)

checks["completed_delivery_times_present"] = (
    pd.to_numeric(
        orders.loc[
            orders["status"].eq("completed"),
            "delivery_minutes"
        ],
        errors="coerce"
    ).notna().all()
)

checks["cancelled_delivery_times_missing"] = (
    pd.to_numeric(
        orders.loc[
            orders["status"].eq("cancelled"),
            "delivery_minutes"
        ],
        errors="coerce"
    ).isna().all()
)

completed_revenue = pd.to_numeric(
    orders.loc[
        orders["status"].eq("completed"),
        "order_amount"
    ],
    errors="raise"
).sum()

checks["completed_revenue"] = (
    round(float(completed_revenue), 2) == 2998668.80
)

# ----------------------------------------------------------
# Check SQLite tables and their contents
# ----------------------------------------------------------

with sqlite3.connect(DB) as connection:
    table_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    checks["sqlite_tables"] = {
        "customers", "products", "orders"
    }.issubset(table_names)

    for table in ("customers", "products", "orders"):
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table}",
            connection
        ).iloc[0]["n"]

        expected = {
            "customers": 1200,
            "products": 40,
            "orders": 12309,
        }[table]

        checks[f"sqlite_{table}_row_count"] = int(count) == expected

    db_revenue = pd.read_sql_query(
        """
        SELECT ROUND(SUM(order_amount), 2) AS revenue
        FROM orders
        WHERE status = 'completed'
        """,
        connection
    ).iloc[0]["revenue"]

    checks["sqlite_completed_revenue"] = (
        round(float(db_revenue), 2) == 2998668.80
    )

# ----------------------------------------------------------
# Display results
# ----------------------------------------------------------

print("VALIDATION RESULTS")

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

failed = [name for name, passed in checks.items() if not passed]

if failed:
    raise AssertionError(
        "Validation failed: " + ", ".join(failed)
    )

print("ALL CHECKPOINTS PASSED")



