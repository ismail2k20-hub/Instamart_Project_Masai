
"""Audit the raw Instamart CSV files."""

import pandas as pd

RAW = "data/raw"

# Load the three main raw tables.
customers = pd.read_csv(f"{RAW}/customers.csv")
products = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")

# Keep blank ticket text as an empty string when tickets are loaded.
tickets = pd.read_csv(
    f"{RAW}/tickets.csv",
    keep_default_na=False
)


def report_table(name, df):
    """Print shape, data types, and missing-value counts."""
    print(f"\n{'=' * 60}")
    print(f"TABLE: {name}")
    print(f"{'=' * 60}")

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })

    print("\nMissing values by column:")
    print(missing_report.to_string())


for name, df in [
    ("customers", customers),
    ("products", products),
    ("orders", orders),
]:
    report_table(name, df)


# -------------------------
# Orders audit figures
# -------------------------

numeric_amount = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

orders_audit = {
    "Raw order rows": len(orders),
    "Exact duplicate rows": int(orders.duplicated().sum()),
    "Pending rows": int(
        orders["status"].eq("pending").sum()
    ),
    "Distinct order IDs": int(
        orders["order_id"].nunique()
    ),
    "Non-numeric order amounts": int(
        numeric_amount.isna().sum()
    ),
    "Negative delivery times": int(
        delivery_numeric.lt(0).sum()
    ),
    "Blank delivery times": int(
        orders["delivery_minutes"].isna().sum()
        + orders["delivery_minutes"].eq("").sum()
    ),
}

# -------------------------
# Customers audit figures
# -------------------------

age_numeric = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

city = customers["city"].fillna("").astype(str).str.strip()
gender = customers["gender"].fillna("").astype(str).str.strip()

customers_audit = {
    "Blank ages": int(
        customers["age"].isna().sum()
        + customers["age"].eq("").sum()
    ),
    "Ages outside 10-100": int(
        age_numeric.lt(10).fillna(False).sum()
        + age_numeric.gt(100).fillna(False).sum()
    ),
    "Distinct raw city spellings": int(
        customers["city"].nunique()
    ),
    "Distinct non-blank raw gender spellings": int(
        gender[gender.ne("")].nunique()
    ),
    "Blank genders": int(
        gender.eq("").sum()
    ),
}

print("\n" + "=" * 60)
print("SPECIFIC AUDIT FIGURES")
print("=" * 60)

for label, value in {
    **orders_audit,
    **customers_audit
}.items():
    print(f"{label}: {value}")
