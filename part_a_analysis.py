from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Keep completed orders only for revenue and spend analysis.
completed = orders.loc[
    orders["status"] == "completed"
].copy()

for column in [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes",
]:
    completed[column] = pd.to_numeric(
        completed[column],
        errors="coerce",
    )

# ------------------------------------------------------------
# 1. PANDAS PIVOT TABLE TO CROSS-CHECK SQL Q2
# ------------------------------------------------------------

q2_data = (
    completed
    .merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    .merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
)

pivot = pd.pivot_table(
    q2_data,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Total",
)

pivot = pivot.round(2)

grand_total = float(pivot.loc["Total", "Total"])

assert round(grand_total, 2) == 2_998_668.80

print("\n=== PANDAS PIVOT TABLE: Q2 CROSS-CHECK (INR) ===")
print(pivot.to_string())
print(f"Pivot grand total: INR {grand_total:,.2f}")
print("SQL Q2 and pandas pivot totals match.")

# ------------------------------------------------------------
# 2. GROUPBY WITH NAMED AGGREGATION
# ------------------------------------------------------------

city_stats = q2_data.groupby("city").agg(
    completed_revenue_inr=("order_amount", "sum"),
    mean_order_amount_inr=("order_amount", "mean"),
    completed_order_count=("order_amount", "count"),
).sort_index()

city_stats["completed_revenue_inr"] = (
    city_stats["completed_revenue_inr"].round(2)
)

city_stats["mean_order_amount_inr"] = (
    city_stats["mean_order_amount_inr"].round(2)
)

assert round(
    float(city_stats.loc["Bengaluru", "completed_revenue_inr"]), 2
) == 523_643.40

assert round(
    float(city_stats.loc["Bengaluru", "mean_order_amount_inr"]), 2
) == 276.47

assert int(
    city_stats.loc["Bengaluru", "completed_order_count"]
) == 1894

assert round(
    float(city_stats.loc["Mumbai", "completed_revenue_inr"]), 2
) == 479_058.50

assert round(
    float(city_stats.loc["Mumbai", "mean_order_amount_inr"]), 2
) == 297.00

assert int(
    city_stats.loc["Mumbai", "completed_order_count"]
) == 1613

print("\n=== COMPLETED ORDER STATISTICS BY CITY ===")
print(city_stats.to_string())

# ------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS AND IQR OUTLIERS
# ------------------------------------------------------------

amount = completed["order_amount"].dropna()

mean_amount = float(amount.mean())
median_amount = float(amount.median())
skew_amount = float(amount.skew())

q1 = float(amount.quantile(0.25))
q3 = float(amount.quantile(0.75))

iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers_below = int((amount < lower_fence).sum())
outliers_above = int((amount > upper_fence).sum())

print("\n=== COMPLETED ORDER AMOUNT DISTRIBUTION ===")
print("Count:", len(amount))
print(f"Mean: INR {mean_amount:,.2f}")
print(f"Median: INR {median_amount:,.2f}")
print(f"Skewness: {skew_amount:.2f}")
print(f"Q1: INR {q1:,.2f}")
print(f"Q3: INR {q3:,.2f}")
print(f"IQR: INR {iqr:,.2f}")
print(f"Lower 1.5 x IQR fence: INR {lower_fence:,.2f}")
print(f"Upper 1.5 x IQR fence: INR {upper_fence:,.2f}")
print("Outliers below lower fence:", outliers_below)
print("Outliers above upper fence:", outliers_above)

assert round(mean_amount, 2) == 269.91
assert round(median_amount, 2) == 135.00
assert round(skew_amount, 2) == 4.61
assert round(q1, 2) == 64.80
assert round(q3, 2) == 297.00
assert round(upper_fence, 2) == 645.30
assert outliers_below == 0
assert outliers_above == 897

# ------------------------------------------------------------
# 4. CORRELATION MATRIX
# ------------------------------------------------------------

corr_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes",
]

corr = completed[corr_columns].corr(numeric_only=True)

print("\n=== CORRELATION MATRIX: COMPLETED ORDERS ===")
print(corr.round(3).to_string())

pairs = []

for i, left in enumerate(corr_columns):
    for right in corr_columns[i + 1:]:
        pairs.append((
            left,
            right,
            float(corr.loc[left, right]),
        ))

largest_pair = max(
    pairs,
    key=lambda item: abs(item[2]),
)

print(
    "Largest absolute correlation: "
    f"{largest_pair[0]} vs {largest_pair[1]} "
    f"= {largest_pair[2]:+.3f}"
)

assert largest_pair[0] == "order_amount"
assert largest_pair[1] == "discount_pct"
assert round(largest_pair[2], 3) == -0.095

print("\nPART A SQL AND PANDAS CHECKS PASSED")