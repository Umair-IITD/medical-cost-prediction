-- View the first 10 rows of the insurance dataset
SELECT * FROM insurance
LIMIT 10;


-- Find any missing values in the dataset
SELECT 
    SUM(age IS NULL) AS missing_age,
    SUM(sex IS NULL) AS missing_sex,
    SUM(bmi IS NULL) AS missing_bmi,
    SUM(children IS NULL) AS missing_children,
    SUM(smoker IS NULL) AS missing_smoker,
    SUM(region IS NULL) AS missing_region,
    SUM(charges IS NULL) AS missing_charges
FROM insurance;


-- Statistical Summary of the dataset
SELECT
	COUNT(*) AS Total_Rows,
    COUNT(DISTINCT age, sex, bmi, children, smoker, region, charges) AS distinct_rows,
    CASE 
        WHEN COUNT(*) = COUNT(DISTINCT age, sex, bmi, children, smoker, region, charges) THEN 'No'
        ELSE 'Yes'
    END AS has_duplicates,
    -- Age
    ROUND(AVG(age),3) AS mean_age,
    MIN(age) AS min_age,
    MAX(age) AS max_age,
    MAX(age) - MIN(age) AS range_age,
    -- BMI
    ROUND(AVG(bmi),3) AS mean_bmi,
    MIN(bmi) AS min_bmi,
    MAX(bmi) AS max_bmi,
    ROUND(MAX(bmi) - MIN(bmi),3) AS range_bmi,
    -- Children
    ROUND(AVG(children),3) AS mean_children,
    MIN(children) AS min_children,
    MAX(children) AS max_children,
    MAX(children) - MIN(children) AS range_children,
    -- Charges
    ROUND(AVG(charges),3) AS mean_charges,
    MIN(charges) AS min_charges,
    MAX(charges) AS max_charges,
    ROUND(MAX(charges) - MIN(charges),3) AS ranges_charges,
    -- Distinct Regions
    COUNT(DISTINCT region) AS distinct_regions,
    -- Distinct Gender
    COUNT(DISTINCT sex) AS distinct_gender,
    -- Boolean Smoker - Yes/No
    COUNT(DISTINCT smoker) AS boolean_smoker
FROM insurance;


-- Check for duplicate rows in the dataset
SELECT 
    age, sex, bmi, children, smoker, region, charges,
    COUNT(*) AS duplicate_count
FROM insurance
GROUP BY age, sex, bmi, children, smoker, region, charges
HAVING COUNT(*) > 1;


-- 1. Categorical Feature Distributions
SELECT sex, COUNT(*) AS count FROM insurance GROUP BY sex;
SELECT smoker, COUNT(*) AS count FROM insurance GROUP BY smoker;
SELECT region, COUNT(*) AS count FROM insurance GROUP BY region;


-- 2. Categorical vs. Target Analysis (Mean Charges)
SELECT smoker, AVG(charges) AS avg_charges FROM insurance GROUP BY smoker;
SELECT sex, AVG(charges) AS avg_charges FROM insurance GROUP BY sex;
SELECT region, AVG(charges) AS avg_charges FROM insurance GROUP BY region;


-- 3. Outlier Detection (IQR Method for Charges)
WITH OrderedData AS (
    SELECT 
        charges,
        ROW_NUMBER() OVER (ORDER BY charges) as rn,
        COUNT(*) OVER () as total_rows
    FROM insurance
),
Quartiles AS (
    SELECT 
        MAX(CASE WHEN rn = FLOOR(total_rows * 0.25) THEN charges END) AS Q1,
        MAX(CASE WHEN rn = FLOOR(total_rows * 0.75) THEN charges END) AS Q3
    FROM OrderedData
),
Bounds AS (
    SELECT 
        Q1, 
        Q3, 
        (Q3 - Q1) AS IQR,
        Q1 - 1.5 * (Q3 - Q1) AS Lower_Bound,
        Q3 + 1.5 * (Q3 - Q1) AS Upper_Bound
    FROM Quartiles
)
SELECT i.* 
FROM insurance i
CROSS JOIN Bounds b
WHERE i.charges < b.Lower_Bound OR i.charges > b.Upper_Bound;