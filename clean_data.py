
"""Clean Instamart customer, product, and order data."""

import os
import sqlite3

import pandas as pd

RAW = "data/raw"
CLEAN = "data/clean"
DB_PATH = "data/instamart.db"

os.makedirs(CLEAN, exist_ok=True)
os.makedirs("data", exist_ok=True)


def read_csv(name, **kwargs):
    return pd.read_csv(os.path.join(RAW, name), **kwargs)


def save_csv(df, name):
    df.to_csv(
        os.path.join(CLEAN, name),
        index=False
    )


# ==========================================================
# 1. LOAD RAW TABLES
# ==========================================================

customers = read_csv("customers.csv")
products = read_csv("products.csv")
orders = read_csv("orders.csv")

# Empty ticket texts must remain empty strings when tickets
# are used in a later analysis.
tickets = read_csv(
    "tickets.csv",
    keep_default_na=False
)

print("Loaded raw tables.")


# ==========================================================
# 2. CLEAN CUSTOMERS
# ==========================================================

customers = customers.copy()

# Rule 1: Standardise city names.
customers["city"] = (
    customers["city"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.title()
)

# Rule 2: Standardise gender values.
gender_map = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
}

gender_raw = (
    customers["gender"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

customers["gender"] = gender_raw.map(gender_map).fillna("unknown")

# Rule 3: Convert age to numeric.
age_raw = customers["age"].replace("", pd.NA)
customers["age"] = pd.to_numeric(
    age_raw,
    errors="coerce"
)

# Ages below 10 or above 100 are invalid.
invalid_age_mask = (
    customers["age"].notna()
    & ~customers["age"].between(10, 100)
)

invalid_age_count = int(invalid_age_mask.sum())
customers.loc[invalid_age_mask, "age"] = pd.NA

# Rule 4: Fill missing ages with their city's median.
# transform("median") returns a value for each row.
age_missing_before_fill = int(customers["age"].isna().sum())

city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)

# The supplied data has enough valid ages in each city.
if customers["age"].isna().any():
    raise ValueError(
        "Some ages remain missing; inspect the city groups."
    )

customers["age"] = customers["age"].round().astype(int)

# Rule 5: Convert signup_date to datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="raise"
)

print("\nCUSTOMER CLEANING")
print("Invalid ages changed to missing:", invalid_age_count)
print("Missing ages filled:", age_missing_before_fill)
print("Cities:", customers["city"].nunique())
print("\nCustomer city counts:")
print(customers["city"].value_counts().sort_index())
print("\nCustomer gender counts:")
print(customers["gender"].value_counts().sort_index())


# ==========================================================
# 3. PREPARE PRODUCTS
# ==========================================================

products = products.copy()

products["price_inr"] = pd.to_numeric(
    products["price_inr"],
    errors="raise"
)


# ==========================================================
# 4. CLEAN ORDERS
# ==========================================================

orders = orders.copy()

# Rule 1: Convert order_ts to datetime.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    format="%Y-%m-%d %H:%M:%S",
    errors="raise"
)

# Rule 2: Remove exact duplicate rows.
rows_before_exact_dedup = len(orders)
orders = orders.drop_duplicates().copy()
exact_duplicates_removed = rows_before_exact_dedup - len(orders)

# Rule 3: Keep the latest record for each order_id.
# mergesort is stable, making sorting deterministic.
rows_before_latest = len(orders)

orders = (
    orders.sort_values(
        ["order_id", "order_ts"],
        kind="mergesort"
    )
    .drop_duplicates(
        subset=["order_id"],
        keep="last"
    )
    .copy()
)

different_record_duplicates_removed = (
    rows_before_latest - len(orders)
)

# Convert numeric columns.
orders["quantity"] = pd.to_numeric(
    orders["quantity"],
    errors="raise"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"],
    errors="raise"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"].replace("", pd.NA),
    errors="coerce"
)

# Rule 4: Fill missing order amounts using product price.
# Merge the catalogue price and category onto each order.
orders = orders.merge(
    products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

missing_amount_count = int(orders["order_amount"].isna().sum())

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
).round(2)

if orders["order_amount"].isna().any():
    raise ValueError("Some order amounts could not be filled.")

# Rule 5: Negative delivery times are invalid.
negative_delivery_mask = orders["delivery_minutes"].lt(0).fillna(False)
negative_delivery_count = int(negative_delivery_mask.sum())

orders.loc[negative_delivery_mask, "delivery_minutes"] = pd.NA

# Cancelled orders were never delivered.
cancelled_mask = orders["status"].eq("cancelled")
orders.loc[cancelled_mask, "delivery_minutes"] = pd.NA

# Impute missing delivery times for completed orders only.
completed_mask = orders["status"].eq("completed")

completed_missing_before = int(
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

if orders.loc[completed_mask, "delivery_minutes"].isna().any():
    raise ValueError(
        "Some completed orders still have missing delivery times."
    )

# Drop the temporary product lookup columns.
orders = orders.drop(columns=["category", "price_inr"])

# Keep a consistent timestamp representation for CSV and SQLite.
orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# Standardise integer-like fields.
orders["quantity"] = orders["quantity"].astype(int)
orders["discount_pct"] = orders["discount_pct"].astype(int)

# Preserve fractional delivery values if a category median is fractional.
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Sort rows for reproducible output.
orders = orders.sort_values("order_id").reset_index(drop=True)

print("\nORDER CLEANING")
print("Exact duplicate rows removed:", exact_duplicates_removed)
print(
    "Additional duplicate order records removed:",
    different_record_duplicates_removed
)
print("Missing order amounts filled:", missing_amount_count)
print("Negative delivery times changed to missing:", negative_delivery_count)
print(
    "Missing completed delivery times before imputation:",
    completed_missing_before
)
print("Final order rows:", len(orders))
print("Final order IDs:", orders["order_id"].nunique())
print("Final order statuses:")
print(orders["status"].value_counts().sort_index())


# ==========================================================
# 5. SAVE CLEAN CSV FILES
# ==========================================================

save_csv(customers, "customers.csv")
save_csv(products, "products.csv")
save_csv(orders, "orders.csv")

print("\nSaved clean CSV files in data/clean/.")


# ==========================================================
# 6. SAVE SQLITE TABLES
# ==========================================================

# Replace the database so rerunning this cleaning script
# cannot leave stale rows or old table definitions.
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

with sqlite3.connect(DB_PATH) as connection:
    customers.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        connection,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        connection,
        if_exists="replace",
        index=False
    )

print("Saved SQLite database:", DB_PATH)


# ==========================================================
# 7. BASIC POST-CLEANING CHECKS
# ==========================================================

assert len(customers) == 1200
assert customers["city"].nunique() == 6
assert customers["gender"].value_counts().to_dict() == {
    "female": 610,
    "male": 560,
    "unknown": 30
}
assert customers["age"].between(18, 60).all()

assert len(products) == 40
assert len(orders) == 12309
assert orders["order_id"].is_unique
assert not orders["status"].eq("pending").any()
assert orders["order_amount"].notna().all()
assert orders.loc[
    orders["status"].eq("completed"),
    "delivery_minutes"
].notna().all()
assert orders.loc[
    orders["status"].eq("cancelled"),
    "delivery_minutes"
].isna().all()

completed_revenue = orders.loc[
    orders["status"].eq("completed"),
    "order_amount"
].sum()

print("Completed revenue INR:", f"{completed_revenue:.2f}")
print("Basic cleaning checks passed.")
