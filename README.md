# Multiple Linear Regression from Scratch: An Empirical and Theoretical Investigation into Medical Cost Prediction

## Abstract

This repository presents an academic, first-principles implementation and econometric evaluation of multiple linear regression designed to predict individual medical insurance expenditures. Standard machine learning implementations typically encapsulate model fitting within automated libraries (such as `scikit-learn`), obscuring the underlying mathematical mechanics. In contrast, this project constructs the complete end-to-end statistical learning workflow from the ground up: relational database schemas and SQL auditing scripts, exploratory data diagnostics, leakage-free data preprocessing, vectorized parameter optimization via batch gradient descent in pure NumPy, systematic hyperparameter exploration, and multi-metric out-of-sample evaluation.

The system is trained and evaluated on the Medical Cost Personal Dataset ($N = 1,338$ records, cleaned to $N = 1,337$ following exact deduplication). Using eight processed predictors—incorporating standardized continuous physiological metrics and dummy-encoded demographic and regional indicators—the optimal model achieves a Coefficient of Determination of **$R^2 = 0.7774$**, a Mean Absolute Error of **$\$4,041.91$**, and a Root Mean Squared Error of **$\$6,162.78$** on an untouched holdout test partition ($N = 201$).

```
+--------------------------------------------------------------------------------------------------+
|                      END-TO-END SYSTEM ARCHITECTURE AND EXECUTION PIPELINE                       |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [ Phase 1: Relational Storage & SQL Auditing ]                                                  |
|  MySQL 8.0 Ingestion -> Nullity Audits -> Exact Deduplication -> IQR Outlier Profiling          |
|         │                                                                                        |
|         ▼                                                                                        |
|  [ Phase 2: Leakage-Free Preprocessing Pipeline ]                                                |
|  70/15/15 Data Split -> Binary Indicator Mapping -> Reference-Omitted Dummy Encoding             |
|  Training-Only Z-Score Standardization -> Serialized float64 NumPy Tensors                       |
|         │                                                                                        |
|         ▼                                                                                        |
|  [ Phase 3: Mathematical Model & First-Principles Vectorization ]                                |
|  Hypothesis: Y_hat = X*W + b -> Half-MSE Objective: J = (1 / 2m) * ||E||_2^2                     |
|  Analytical Gradients: grad_W = (1 / m)*X^T*E, grad_b = (1 / m)*1^T*E                            |
|         │                                                                                        |
|         ▼                                                                                        |
|  [ Phase 4: Systematic Learning Rate Optimization Sweep ]                                        |
|  Grid: alpha in {0.0001, 0.001, 0.003, 0.01, 0.1, 0.2, 0.3, 1.0, 1.3}                          |
|  Monotonic Convergence at alpha = 0.3 -> Checkpoint Stability -> Optimal W_best, b_best          |
|         │                                                                                        |
|         ▼                                                                                        |
|  [ Phase 5: Comprehensive Out-of-Sample Model Evaluation ]                                       |
|  Triple Partition Appraisal (Train/Val/Test) -> MAE, RMSE, R^2 -> Residual Diagnostics          |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

---

## 1. Executive Summary and Empirical Benchmark Findings

The primary predictive engine was optimized using batch gradient descent over $T = 1,000$ iterations with an optimal learning rate of $\alpha = 0.3$. Model parameters were estimated strictly on the training partition ($N_{\text{train}} = 935$), tuned on the validation partition ($N_{\text{val}} = 201$), and benchmarked on an independent holdout test partition ($N_{\text{test}} = 201$).

**Table 1: Final Cross-Partition Benchmark Performance Metrics**

| Empirical Metric | Formula / Criterion | Training ($N = 935$) | Validation ($N = 201$) | Holdout Test ($N = 201$) | Interpretive Units |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Half-MSE Objective ($J$)** | $\frac{1}{2m} \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ | $17,906,466.62$ | $19,950,234.93$ | **$18,989,836.43$** | $\text{USD}^2$ |
| **Mean Absolute Error (MAE)** | $\frac{1}{m} \sum \|y^{(i)} - \hat{y}^{(i)}\|$ | $\$4,165.83$ | $\$4,321.72$ | **$\$4,041.91$** | $\text{USD}$ (\$) |
| **Root Mean Squared Error (RMSE)**| $\sqrt{\frac{1}{m} \sum (y^{(i)} - \hat{y}^{(i)})^2}$ | $\$5,984.39$ | $\$6,316.68$ | **$\$6,162.78$** | $\text{USD}$ (\$) |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | $0.7363$ ($73.63\%$) | $0.7668$ ($76.68\%$) | **$0.7774$ ($77.74\%$)** | Dimensionless |
| **Error Spread ($\text{RMSE} - \text{MAE}$)**| Error Magnitude Dispersion | $\$1,818.56$ | $\$1,994.96$ | **$\$2,120.87$** | $\text{USD}$ (\$) |

### 1.1 Final Learned Parameters

The final model parameters estimated by first-principles gradient descent ($\alpha = 0.3, T = 1,000$) are:
- **Scalar Intercept**: $b = 8,905.4513$
- **Learned Weight Vector ($\mathbf{W} \in \mathbb{R}^{8 \times 1}$)**:
  - $w_{\text{age}} = +3,515.4201$ ($+\$251.25$ per additional year of age)
  - $w_{\text{sex}} = -111.9070$ (minimal conditional difference between biological sexes)
  - $w_{\text{bmi}} = +1,977.1322$ ($+\$328.38$ per $\text{kg/m}^2$ of Body Mass Index)
  - $w_{\text{children}} = +625.1713$ ($+\$522.15$ per dependent child covered)
  - $w_{\text{smoker}} = +22,874.4465$ (massive surcharge for tobacco consumption)
  - $w_{\text{region\_northwest}} = -465.7452$ (relative to Northeast baseline)
  - $w_{\text{region\_southeast}} = -936.0979$ (relative to Northeast baseline)
  - $w_{\text{region\_southwest}} = -765.5825$ (relative to Northeast baseline)

---

## 2. Theoretical Formulation and Mathematical Architecture

### 2.1 The Vectorized Affine Hypothesis

For a design matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ comprising $m$ instances and $n = 8$ predictors, the hypothesis computes:

$$
\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + b\mathbf{1}_m
$$

where $\mathbf{W} \in \mathbb{R}^{n \times 1}$ is the coefficient vector, $b \in \mathbb{R}$ is the scalar bias, and $\mathbf{1}_m$ represents the broadcasting of the intercept across all observations.

### 2.2 Objective Loss Function: Convex Half-MSE

The empirical risk is quantified via the Half-Mean-Squared-Error cost function $J(\mathbf{W}, b)$:

$$
J(\mathbf{W}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 = \frac{1}{2m} \|\mathbf{E}\|_2^2 = \frac{1}{2m} \mathbf{E}^T \mathbf{E}
$$

where $\mathbf{E} = \hat{\mathbf{Y}} - \mathbf{Y} \in \mathbb{R}^{m \times 1}$ is the residual error vector.

The constant $\frac{1}{2}$ cancels out when differentiating the quadratic error. Because the design matrix exhibits full column rank, the Hessian matrix $\nabla^2 J = \frac{1}{m} \mathbf{X}^T \mathbf{X}$ is strictly positive definite, proving strict convexity and guaranteeing convergence to a unique global minimum.

### 2.3 Analytic Vector Calculus Gradients

Differentiating the scalar cost function with respect to weights and bias yields the gradient expressions:

$$
\nabla_{\mathbf{W}} J = \frac{\partial J}{\partial \mathbf{W}} = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{Y}} - \mathbf{Y}) = \frac{1}{m} \mathbf{X}^T \mathbf{E} \in \mathbb{R}^{n \times 1}
$$

$$
\nabla_b J = \frac{\partial J}{\partial b} = \frac{1}{m} \mathbf{1}_m^T (\hat{\mathbf{Y}} - \mathbf{Y}) = \frac{1}{m} \sum_{i=1}^{m} e^{(i)} \in \mathbb{R}
$$

### 2.4 Batch Gradient Descent Parameter Updates

Parameters are updated along the negative gradient vector at each epoch $t$:

$$
\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \alpha \nabla_{\mathbf{W}} J^{(t)} = \mathbf{W}^{(t)} - \frac{\alpha}{m} \mathbf{X}^T \mathbf{E}^{(t)}
$$

$$
b^{(t+1)} = b^{(t)} - \alpha \nabla_b J^{(t)} = b^{(t)} - \frac{\alpha}{m} \sum_{i=1}^{m} e^{(i)(t)}
$$

---

## 3. Repository Architecture and Directory Topology

The project is structured into modular components separating data, SQL queries, source modules, notebook workflows, generated figures, and academic reports:

```text
Multiple Linear Regression - Medical Cost Prediction/
├── data/
│   ├── raw/
│   │   └── insurance.csv                         # Primary observational microdata (1,338 rows)
│   └── processed/                                # Destination directory for serialized partitions
├── sql/
│   ├── 01_data_load.sql                          # MySQL relational schema & raw ingestion DDL
│   └── 02_data_analysis.sql                      # Integrity, summary statistics, deduplication, IQR audits
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb        # Comprehensive interactive research notebook
├── src/
│   ├── __init__.py                               # Python package identifier
│   ├── data_preprocessing.py                     # Data cleaning, 70/15/15 split, encoding, scaling
│   ├── linear_regression.py                      # Pure NumPy hypothesis, cost, gradients, batch descent
│   ├── metrics_calculation.py                    # Metric implementations: MAE, RMSE, R-squared
│   └── visualization.py                          # Publication-grade plotting routines
├── outputs/
│   ├── figures/                                  # Empirical distributions, scatter plots, residual diagnostics
│   │   ├── raw_charges_distribution.png
│   │   ├── raw_charges_vs_age.png
│   │   ├── raw_charges_vs_bmi.png
│   │   ├── raw_charges_by_smoker.png
│   │   ├── raw_charges_by_region.png
│   │   ├── scaled_age_distribution.png
│   │   ├── scaled_bmi_distribution.png
│   │   ├── scaled_children_distribution.png
│   │   ├── processed_charges_vs_age.png
│   │   ├── processed_charges_vs_bmi.png
│   │   ├── final_gradient_descent_learning_curve.png
│   │   ├── actual_vs_predicted_insurance_charges.png
│   │   └── residual_plots.png
│   └── learning_curves/                          # Optimization trajectories across all learning rates
│       ├── Learning Curve for alpha = 0.0001.png
│       ├── Learning Curve for alpha = 0.001.png
│       ├── Learning Curve for alpha = 0.003.png
│       ├── Learning Curve for alpha = 0.01.png
│       ├── Learning Curve for alpha = 0.1.png
│       ├── Learning Curve for alpha = 0.2.png
│       ├── Learning Curve for alpha = 0.3.png
│       ├── Learning Curve for alpha = 1.png
│       ├── Learning Curve for alpha = 1.3.png
│       └── final_learning_rate_comparison.png
├── reports/
│   ├── 01_data_analysis_and_preprocessing.md     # Chapter 1: Exploratory Analysis & Preprocessing
│   ├── 02_multiple_linear_regression_model.md    # Chapter 2: Mathematical Formulations & Derivations
│   ├── 03_learning_rate_experiments.md           # Chapter 3: Optimization Dynamics & Hyperparameters
│   └── 04_final_results_and_evaluation.md        # Chapter 4: Model Evaluation & Residual Diagnostics
├── requirements.txt                              # Pinned Python package dependencies
├── .gitignore                                    # VCS tracking exclusions
└── README.md                                     # Comprehensive Project Monograph
```

---

## 4. Comprehensive Thesis Reports Roadmap

The technical reports in `reports/` form a four-chapter monograph detailing the project methodology and findings:

1. **[Chapter 1: Exploratory Data Analysis, Statistical Auditing, and Rigorous Preprocessing Methodology](reports/01_data_analysis_and_preprocessing.md)**
   - Schema definitions, attribute topologies, and domain descriptions.
   - MySQL database integration, completeness audits (verifying zero missing values), and duplicate detection (identifying the single duplicate row).
   - Descriptive summary statistics, target distribution analysis, and Tukey's IQR outlier auditing ($139$ high-cost cases retained for clinical validity).
   - Data partitioning ($70\%$ train, $15\%$ validation, $15\%$ test), binary indicator mapping, dummy encoding with Northeast omitted to avoid the dummy variable trap, and training-only continuous standardization ($\mu, \sigma$).
   - NumPy tensor serialization (`float64`) and matrix shape validation (`(935, 8)` and `(935, 1)`).

2. **[Chapter 2: Multiple Linear Regression — Theoretical Foundations, Mathematical Derivations, and First-Principles Implementation](reports/02_multiple_linear_regression_model.md)**
   - Theoretical framing under the Gauss-Markov theorem and empirical risk minimization.
   - Formulation of the affine hypothesis $\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + b\mathbf{1}_m$.
   - Mathematical derivation of the Half-MSE objective function $J(\mathbf{W}, b)$ and formal proof of cost surface convexity via the positive-definite Hessian $\nabla^2 J = \frac{1}{m}\mathbf{X}^T\mathbf{X}$.
   - Analytic gradient derivations via the chain rule: $\nabla_{\mathbf{W}} J = \frac{1}{m}\mathbf{X}^T\mathbf{E}$ and $\nabla_b J = \frac{1}{m}\sum e_i$.
   - Batch gradient descent pseudocode, implementation review, and actuarial interpretation of learned weights.

3. **[Chapter 3: Optimization Dynamics, Learning Rate Hyperparameter Exploration, and Convergence Diagnostics](reports/03_learning_rate_experiments.md)**
   - Theoretical convergence limits based on the Lipschitz constant: $\alpha < \frac{2}{\lambda_{\max}(\mathbf{H})}$.
   - Empirical sweep across nine learning rates: $\alpha \in \{0.0001, 0.001, 0.003, 0.01, 0.1, 0.2, 0.3, 1.0, 1.3\}$ over $T = 1,000$ iterations.
   - Analysis of four optimization regimes: under-stepping ($\alpha \le 0.01$), optimal monotonic descent ($\alpha \in [0.1, 0.3]$), oscillatory overshoot ($\alpha = 1.0$), and numerical divergence ($\alpha = 1.3$, where cost explodes to $\approx 5.54 \times 10^{47}$).
   - Multi-criteria validation comparison justifying the selection of **$\alpha = 0.3$** based on validation loss and checkpoint monotonicity.

4. **[Chapter 4: Comprehensive Model Evaluation, Residual Diagnostics, and Empirical Findings](reports/04_final_results_and_evaluation.md)**
   - Mathematical formulations of MAE, RMSE, and $R^2$, including proof of $\text{RMSE} \ge \text{MAE}$ via Jensen's Inequality.
   - Cross-partition performance evaluation on the native dollar scale ($N_{\text{test}} = 201$: $\text{MAE} = \$4,041.91$, $\text{RMSE} = \$6,162.78$, $R^2 = 0.7774$).
   - Actual-versus-predicted response analysis, identifying three risk strata and characterizing high-charge underprediction.
   - Residual diagnostics evaluating mean-zero errors, heteroscedastic fan-shaped dispersion, and structured non-linear banding.
   - Methodological recommendations for future work, including logarithmic response transformations, explicit interaction terms ($\text{BMI} \times \text{smoker}$), and Generalized Linear Models.

---

## 5. Dataset Schema and Actuarial Attributes

**Table 2: Complete Dataset Schema and Operational Definitions**

| Column Name | Storage Type | Statistical Role | Permissible Range | Actuarial Definition & Semantic Scope |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Integer (`int64`) | Continuous Predictor ($X_1$) | $[18, 64] \cap \mathbb{Z}^+$ | Chronological age of the primary insurance beneficiary in years. |
| `sex` | String (`str`) | Binary Dichotomy ($X_2$) | `{"female", "male"}` | Biological sex of beneficiary; binary-encoded (`female` $= 0$, `male` $= 1$). |
| `bmi` | Float (`float64`) | Continuous Predictor ($X_3$) | $[15.96, 53.13] \subset \mathbb{R}^+$ | Body Mass Index ($\text{kg/m}^2$); clinical obesity threshold is $\text{BMI} \ge 30.0$. |
| `children` | Integer (`int64`) | Metric Discrete ($X_4$) | $\{0, 1, 2, 3, 4, 5\}$ | Number of dependent children/dependents covered under the policy. |
| `smoker` | String (`str`) | Binary Dichotomy ($X_5$) | `{"no", "yes"}` | Active tobacco smoking status; binary-encoded (`no` $= 0$, `yes` $= 1$). |
| `region` | String (`str`) | Polytomous Factor ($X_6$) | `{"northeast", "northwest", "southeast", "southwest"}` | Beneficiary residential geographic quadrant; dummy-encoded with Northeast omitted. |
| `charges` | Float (`float64`) | Continuous Target ($Y$) | $[\$1,121.87, \$63,770.43]$ | Total annual medical insurance claims billed to the insurer (in USD \$). |

---

## 6. Visual Analytics Catalog

The repository includes visual artifacts generated by `src/visualization.py`:

**Table 3: Output Visual Artifacts Catalog**

| Output File Path | Functional Description & Analytical Focus |
| :--- | :--- |
| `outputs/figures/raw_charges_distribution.png` | Density estimation and histogram illustrating right-skewed medical claims. |
| `outputs/figures/raw_charges_vs_age.png` | Scatter plot of charges vs. age stratified by smoking, revealing three parallel cost bands. |
| `outputs/figures/raw_charges_vs_bmi.png` | Scatter plot of charges vs. BMI, highlighting the cost surge for obese smokers ($\text{BMI} \ge 30.0$). |
| `outputs/figures/raw_charges_by_smoker.png` | Boxplot comparing charges across smoking cohorts, showing a $\approx \$23,600$ difference in means. |
| `outputs/figures/raw_charges_by_region.png` | Boxplot comparing charges across the four US geographic quadrants. |
| `outputs/figures/scaled_age_distribution.png` | Distribution of standardized age ($Z$-scores) on the training set. |
| `outputs/figures/scaled_bmi_distribution.png` | Distribution of standardized BMI ($Z$-scores), exhibiting a symmetric bell-shaped profile. |
| `outputs/figures/scaled_children_distribution.png` | Discrete probability distribution of standardized dependent child counts. |
| `outputs/figures/processed_charges_vs_age.png` | Unscaled insurance charges plotted against standardized age. |
| `outputs/figures/processed_charges_vs_bmi.png` | Unscaled insurance charges plotted against standardized BMI. |
| `outputs/figures/final_gradient_descent_learning_curve.png` | Convergence curve for the selected model ($\alpha = 0.3$) over $1,000$ iterations. |
| `outputs/figures/actual_vs_predicted_insurance_charges.png` | Actual vs. predicted charges for training and validation sets, showing three risk clusters. |
| `outputs/figures/residual_plots.png` | Residual scatter plots ($e$ vs. $\hat{Y}$) diagnosing heteroscedasticity and non-linear banding. |
| `outputs/learning_curves/final_learning_rate_comparison.png` | Comparative convergence curves across candidate learning rates $\alpha \in \{0.1, 0.2, 0.3, 1.0\}$. |

---

## 7. Environment Setup, Installation, and Execution Guide

### 7.1 Prerequisites
- Python 3.10+ (tested on Python 3.10 through 3.12).
- MySQL Server 8.0+ (optional; required only if re-executing relational database scripts in `sql/`).

### 7.2 Virtual Environment Initialization

From the repository root, create and activate a dedicated virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install pinned dependencies
pip install -r requirements.txt
```

```bash
# macOS / Linux Bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 7.3 Executing the Interactive Research Notebook

The primary workflow is orchestrated within `notebooks/01_exploratory_data_analysis.ipynb`:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

The notebook resolves paths dynamically relative to the repository root, loads the raw dataset, executes SQL-equivalent audit checks, runs the preprocessing pipeline, performs the learning-rate grid sweep, logs convergence curves, and computes evaluation metrics across all partitions.

### 7.4 Running the SQL Audit Layer (Optional)

To execute the database audit scripts independently in MySQL:
1. Ensure the MySQL daemon is active.
2. Edit `sql/01_data_load.sql` to point the `LOAD DATA INFILE` path to the absolute path of `insurance.csv` within your system's secure upload directory (`secure_file_priv`).
3. Execute the SQL scripts:

```bash
mysql -u root -p < sql/01_data_load.sql
mysql -u root -p < sql/02_data_analysis.sql
```

The Python regression pipeline reads `data/raw/insurance.csv` directly via pandas and does not require an active MySQL instance to run.

---

## 8. Reproducibility Protocols and Scientific Rigor

To ensure experimental reproducibility, the codebase implements several core protocols:
1. **Fixed Pseudorandom Seeds**: The dataset partitioning step in `src/data_preprocessing.py` uses `random_state=42` across all splits, ensuring deterministic sample allocation.
2. **Strict Anti-Leakage Discipline**: Scaling parameters ($\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}}$) are fitted exclusively on the 935 training rows and applied downstream to validation and test sets without re-estimation.
3. **Test Partition Isolation**: The test set ($N = 201$) is withheld throughout preprocessing, gradient descent optimization, and hyperparameter tuning, and is evaluated exactly once to establish generalization performance.
4. **Transparent Loss Metrics**: The Half-MSE cost $J(\mathbf{W}, b)$ is tracked in native squared-dollar units ($\$^2$) for optimization, while MAE and RMSE provide dollar-denominated error metrics for evaluation.

---

## 9. Data Attribution and Licensing

The microdata in `data/raw/insurance.csv` originates from the Medical Cost Personal Dataset, derived from demographic and health spending statistics published by the United States Census Bureau and the Centers for Medicare & Medicaid Services (CMS). This dataset is made available for academic research, education, and statistical benchmarking.

The source code in this repository is licensed under the MIT License. Users may freely adapt, extend, and deploy the code provided appropriate attribution is maintained.
