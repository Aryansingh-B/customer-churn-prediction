-- =============================================================
-- churn_analysis.sql
-- Customer Churn Analysis — SQL Queries
-- Database: SQLite (telecom_churn.db)
-- =============================================================


-- ─────────────────────────────────────────────────────────────
-- 1. OVERVIEW: Overall churn rate
-- ─────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                        AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                               AS churn_rate_pct
FROM customers;


-- ─────────────────────────────────────────────────────────────
-- 2. CHURN BY CONTRACT TYPE
--    Business insight: Month-to-month contracts are highest risk
-- ─────────────────────────────────────────────────────────────
SELECT
    Contract,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)            AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                          AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────
-- 3. CHURN BY TENURE SEGMENT
--    Business insight: New customers (< 12m) churn most
-- ─────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12 months (New)'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months (Growing)'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months (Loyal)'
        ELSE '49+ months (Champion)'
    END AS tenure_segment,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)            AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                          AS churn_rate_pct
FROM customers
GROUP BY tenure_segment
ORDER BY churn_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────
-- 4. CHURN BY PAYMENT METHOD
--    Business insight: Electronic check users churn more
-- ─────────────────────────────────────────────────────────────
SELECT
    PaymentMethod,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)            AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                          AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────
-- 5. AVERAGE MONTHLY CHARGES: Churned vs Retained
-- ─────────────────────────────────────────────────────────────
SELECT
    Churn,
    ROUND(AVG(MonthlyCharges), 2)  AS avg_monthly_charges,
    ROUND(MIN(MonthlyCharges), 2)  AS min_monthly_charges,
    ROUND(MAX(MonthlyCharges), 2)  AS max_monthly_charges,
    COUNT(*)                       AS customer_count
FROM customers
GROUP BY Churn;


-- ─────────────────────────────────────────────────────────────
-- 6. HIGH-VALUE CUSTOMERS AT RISK
--    Churned customers with MonthlyCharges > $75
-- ─────────────────────────────────────────────────────────────
SELECT
    customerID,
    Contract,
    tenure,
    MonthlyCharges,
    PaymentMethod,
    InternetService
FROM customers
WHERE Churn = 'Yes'
  AND MonthlyCharges > 75
ORDER BY MonthlyCharges DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────
-- 7. INTERNET SERVICE × CONTRACT: CHURN HEATMAP DATA
-- ─────────────────────────────────────────────────────────────
SELECT
    InternetService,
    Contract,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)            AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                          AS churn_rate_pct
FROM customers
GROUP BY InternetService, Contract
ORDER BY churn_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────
-- 8. SENIOR CITIZEN CHURN ANALYSIS
-- ─────────────────────────────────────────────────────────────
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS customer_type,
    COUNT(*)                                                          AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                   AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                                 AS churn_rate_pct
FROM customers
GROUP BY SeniorCitizen;


-- ─────────────────────────────────────────────────────────────
-- 9. REVENUE AT RISK (Monthly revenue from churned customers)
-- ─────────────────────────────────────────────────────────────
SELECT
    SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) AS monthly_revenue_at_risk,
    SUM(MonthlyCharges)                                          AS total_monthly_revenue,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END)
              / SUM(MonthlyCharges), 2
    )                                                            AS revenue_at_risk_pct
FROM customers;


-- ─────────────────────────────────────────────────────────────
-- 10. CUSTOMER SEGMENTATION FOR RETENTION CAMPAIGNS
--     Priority segments for marketing intervention
-- ─────────────────────────────────────────────────────────────
SELECT
    customerID,
    Contract,
    tenure,
    MonthlyCharges,
    InternetService,
    PaymentMethod,
    CASE
        WHEN Contract = 'Month-to-month' AND tenure < 12 AND MonthlyCharges > 60
            THEN 'PRIORITY-1: High Risk + High Value'
        WHEN Contract = 'Month-to-month' AND tenure < 24
            THEN 'PRIORITY-2: High Risk'
        WHEN MonthlyCharges > 75 AND tenure < 24
            THEN 'PRIORITY-3: High Value, Watch'
        ELSE 'PRIORITY-4: Standard Monitoring'
    END AS retention_priority
FROM customers
WHERE Churn = 'No'  -- Target retained customers BEFORE they churn
ORDER BY
    CASE
        WHEN Contract = 'Month-to-month' AND tenure < 12 THEN 1
        WHEN Contract = 'Month-to-month' THEN 2
        ELSE 3
    END,
    MonthlyCharges DESC;
