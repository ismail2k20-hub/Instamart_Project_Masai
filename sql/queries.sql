-- Instamart Customer Intelligence: Part A
-- Revenue and spend include completed orders only unless specified otherwise.


-- Q1: WHERE, GROUP BY and HAVING
-- Find customers whose completed orders above INR 500
-- add up to more than INR 8,000.

SELECT
    customer_id,
    SUM(order_amount) AS qualifying_order_total_inr
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_order_total_inr DESC, customer_id;


-- Q2: Three-table JOIN
-- Calculate completed revenue by customer city and product category.

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders AS o
JOIN customers AS c
    ON c.customer_id = o.customer_id
JOIN products AS p
    ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Scalar subquery
-- Find products priced above the average catalogue price.

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC, product_name;


-- Q4: LEFT JOIN
-- Find customers who have no completed orders.
-- Keep the completed condition in ON so customers without a match remain.

SELECT
    c.customer_id,
    c.city,
    c.signup_date
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Common Table Expressions (CTEs)
-- Calculate spend for customers with completed orders,
-- then find those whose spend is above the average customer spend.

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT
        AVG(total_spend_inr) AS avg_spend_inr
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend_inr, 2) AS total_spend_inr,
    ROUND(av.avg_spend_inr, 2) AS average_spend_inr
FROM customer_spend AS cs
CROSS JOIN average_spend AS av
WHERE cs.total_spend_inr > av.avg_spend_inr
ORDER BY cs.total_spend_inr DESC, cs.customer_id;


-- Q6: Window functions
-- Rank customer completed spend within each city.

WITH customer_spend AS (
    SELECT
        c.city,
        c.customer_id,
        SUM(o.order_amount) AS total_spend_inr
    FROM customers AS c
    JOIN orders AS o
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT
    city,
    customer_id,
    ROUND(total_spend_inr, 2) AS total_spend_inr,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend_inr DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend_inr DESC
    ) AS spend_dense_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend_inr DESC
    ) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;


-- Q7: CTE and running total window function
-- Calculate daily completed revenue and cumulative revenue by date.

WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue_inr, 2) AS revenue_inr,
    ROUND(
        SUM(revenue_inr) OVER (ORDER BY day),
        2
    ) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY day;