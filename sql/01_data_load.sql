CREATE DATABASE IF NOT EXISTS ML_training_data;

USE ML_training_data;

CREATE TABLE IF NOT EXISTS insurance (
age INT,
sex VARCHAR(10),
bmi FLOAT,
children INT,
smoker VARCHAR(10),
region VARCHAR(20),
charges FLOAT
);

-- Load data from CSV file into the insurance table
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/insurance.csv'
INTO TABLE insurance
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(age, sex, bmi, children, smoker, region, charges);