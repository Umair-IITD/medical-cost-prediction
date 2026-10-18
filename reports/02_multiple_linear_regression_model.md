# Chapter 2: Multiple Linear Regression — Theoretical Foundations, Mathematical Derivations, and First-Principles Implementation

## 1. Abstract and Theoretical Scope

Multiple linear regression represents one of the foundational parametric algorithms in statistical learning. While contemporary machine learning often relies on high-level libraries that encapsulate model fitting within black-box interfaces, constructing an estimator from first principles illuminates the intimate connection between statistical theory, linear algebra, vector calculus, and numerical optimization.

This chapter presents the mathematical formulation, formal differential derivations, and vectorized implementation of a multiple linear regression model built strictly using NumPy. Beginning with the classical Gauss-Markov linear hypothesis, we formalize empirical risk minimization under the half-mean-squared-error ($L_2$) loss function. We derive the analytical gradient vectors for both the weight matrix $\mathbf{W}$ and scalar intercept $b$ using matrix calculus, prove the convexity of the objective surface, and formalize the batch gradient descent optimization algorithm. Finally, we analyze the resulting model coefficients, interpreting their actuarial significance within the context of the medical cost prediction problem.

```
+--------------------------------------------------------------------------------------------------+
|                    MULTIPLE LINEAR REGRESSION FIRST-PRINCIPLES ARCHITECTURE                      |
+--------------------------------------------------------------------------------------------------+
|  Design Matrix X in R^(m x n)               Weight Vector W in R^(n x 1), Bias b in R            |
|         │                                                 │                                      |
|         └───────────────────────┬─────────────────────────┘                                      |
|                                 ▼                                                                |
|                 Forward Pass: Y_hat = X * W + b                                                  |
|                                 │                                                                |
|                                 ▼                                                                |
|                 Residual Vector: E = Y_hat - Y in R^(m x 1)                                      |
|                                 │                                                                |
|         ┌───────────────────────┴─────────────────────────┐                                      |
|         ▼                                                 ▼                                      |
|  Cost Calculation:                               Analytical Gradients:                           |
|  J(W,b) = (1 / 2m) * (E^T * E)                   grad_W = (1 / m) * (X^T * E)                    |
|                                                  grad_b = (1 / m) * sum(E)                       |
|                                                           │                                      |
|                                                           ▼                                      |
|                                          Batch Parameter Update:                                 |
|                                          W := W - alpha * grad_W                                 |
|                                          b := b - alpha * grad_b                                 |
|                                                           │                                      |
|                                                           ▼                                      |
|                                          Iterate for T Epochs until Converged                    |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation of the Multiple Linear Regression Hypothesis

### 2.1 The Scalar Model Equation

In multiple linear regression, the conditional expectation of a continuous response variable $y^{(i)} \in \mathbb{R}$ is modeled as an affine linear combination of $n$ input covariates $x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i)}$:

$$
\hat{y}^{(i)} = b + \sum_{j=1}^{n} w_j x_j^{(i)}
$$

where:
- $\hat{y}^{(i)}$ denotes the model's predicted response for the $i$-th observational unit.
- $w_j \in \mathbb{R}$ represents the learned regression weight (partial slope coefficient) associated with feature $j$.
- $b \in \mathbb{R}$ denotes the scalar intercept (bias term), representing the expected value of $\hat{y}$ when all input features equal zero ($x_j = 0 \; \forall j$).
- $n = 8$ denotes the total number of processed predictors in the design matrix.

### 2.2 Vectorized Matrix Formulation

For a dataset containing $m$ observations, evaluating the scalar equation independently across samples incurs substantial computational overhead. Vectorized matrix notation expresses the hypothesis simultaneously for all $m$ instances:

$$
\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + b\mathbf{1}_m
$$

where:
- $\mathbf{X} \in \mathbb{R}^{m \times n}$ is the design matrix, wherein row $i$ represents the transposed covariate vector $\left(\mathbf{x}^{(i)}\right)^T$:

  $$
  \mathbf{X} = \begin{bmatrix}
  x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
  x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
  \vdots & \vdots & \ddots & \vdots \\
  x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
  \end{bmatrix}
  $$

- $\mathbf{W} \in \mathbb{R}^{n \times 1}$ is the column vector of partial regression coefficients:

  $$
  \mathbf{W} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}
  $$

- $b \in \mathbb{R}$ is the scalar bias term.
- $\mathbf{1}_m \in \mathbb{R}^{m \times 1}$ is a column vector of ones, representing the broadcasting of the scalar intercept across all $m$ observational units.
- $\hat{\mathbf{Y}} \in \mathbb{R}^{m \times 1}$ is the column vector of model predictions.

**Table 2.1: Formal Algebraic Dimensions and Tensor Specifications**

| Mathematical Entity | Symbol | Formal Algebraic Space | NumPy Tensor Shape | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Design Matrix** | $\mathbf{X}$ | $\mathbb{R}^{m \times n}$ | `(935, 8)` | Training partition feature matrix |
| **Weight Vector** | $\mathbf{W}$ | $\mathbb{R}^{n \times 1}$ | `(8, 1)` | Linear regression weight vector |
| **Scalar Intercept** | $b$ | $\mathbb{R}$ | Scalar (`float64`) | Constant additive bias |
| **Target Vector** | $\mathbf{Y}$ | $\mathbb{R}^{m \times 1}$ | `(935, 1)` | Actual unscaled insurance charges |
| **Prediction Vector** | $\hat{\mathbf{Y}}$ | $\mathbb{R}^{m \times 1}$ | `(935, 1)` | Estimated insurance charges |
| **Error / Residual Vector** | $\mathbf{E}$ | $\mathbb{R}^{m \times 1}$ | `(935, 1)` | Difference vector ($\hat{\mathbf{Y}} - \mathbf{Y}$) |

### 2.3 Implementation of the Hypothesis Function

In `src/linear_regression.py`, the forward prediction pass is implemented via `predict()`:

```python
def predict(X, W, b):
    """
    Compute predictions for a given input matrix X using weights W and bias b.
    
    Parameters:
        X: ndarray of shape (m, n)
        W: ndarray of shape (n, 1)
        b: float (scalar bias)
        
    Returns:
        Y_predict: ndarray of shape (m, 1)
    """
    Y_predict = np.matmul(X, W) + b
    return Y_predict
```

By explicitly enforcing two-dimensional tensors (`(m, n)` and `(n, 1)`), the matrix product `np.matmul(X, W)` produces a strict column vector of shape `(m, 1)`. NumPy's broadcasting semantics then add the scalar $b$ across all rows, avoiding rank-1 array `(m,)` broadcasting mismatches.

---

## 3. Loss Function and Empirical Risk Minimization

### 3.1 The Half-Mean-Squared-Error Objective Function

Parameter estimation in linear regression proceeds via empirical risk minimization. The standard Ordinary Least Squares (OLS) criterion minimizes the Mean Squared Error (MSE). In gradient-based machine learning, it is customary to scale the MSE by a factor of $\frac{1}{2}$, yielding the **Half-MSE cost function** $J(\mathbf{W}, b)$:

$$
J(\mathbf{W}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

Expressing the individual observation errors as an error vector $\mathbf{E} \in \mathbb{R}^{m \times 1}$:

$$
\mathbf{E} = \hat{\mathbf{Y}} - \mathbf{Y} = \mathbf{X}\mathbf{W} + b\mathbf{1}_m - \mathbf{Y}
$$

the objective function can be written in quadratic matrix form:

$$
J(\mathbf{W}, b) = \frac{1}{2m} \|\mathbf{E}\|_2^2 = \frac{1}{2m} \mathbf{E}^T \mathbf{E}
$$

### 3.2 Mathematical Justification of the Half-Scale Multiplier

The constant factor $\frac{1}{2}$ is introduced purely for analytical convenience. When computing the partial derivatives of the cost function with respect to the parameters, the exponent $2$ from the squared norm brings down a multiplicative factor of $2$, which cancels out cleanly against $\frac{1}{2}$:

$$
\frac{\partial}{\partial z} \left( \frac{1}{2} z^2 \right) = z
$$

This eliminates redundant scalar constants in the gradient expressions. The relationship between the optimization cost $J(\mathbf{W}, b)$ and conventional evaluation metrics is exact:

$$
\text{MSE} = \frac{1}{m} \|\mathbf{E}\|_2^2 = 2 \cdot J(\mathbf{W}, b)
$$

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{2 \cdot J(\mathbf{W}, b)}
$$

Because the target variable $\mathbf{Y}$ is measured in United States Dollars (\$), the cost $J(\mathbf{W}, b)$ possesses physical units of **squared dollars ($\$^2$)**. Consequently, $J$ serves as an optimization metric and should not be confused with dollar-denominated error metrics such as MAE or RMSE.

### 3.3 Convexity Analysis of the Cost Surface

A critical property of the Half-MSE cost function is its strict convexity over the parameter space $\mathbb{R}^{n+1}$. 

#### Proof of Convexity
Let the parameter vector be augmented as $\tilde{\mathbf{W}} = \begin{bmatrix} b \\ \mathbf{W} \end{bmatrix} \in \mathbb{R}^{n+1}$, and the design matrix be augmented as $\tilde{\mathbf{X}} = [\mathbf{1}_m, \mathbf{X}] \in \mathbb{R}^{m \times (n+1)}$. The cost function becomes:

$$
J(\tilde{\mathbf{W}}) = \frac{1}{2m} (\tilde{\mathbf{X}}\tilde{\mathbf{W}} - \mathbf{Y})^T (\tilde{\mathbf{X}}\tilde{\mathbf{W}} - \mathbf{Y})
$$

Expanding the quadratic form:

$$
J(\tilde{\mathbf{W}}) = \frac{1}{2m} \left( \tilde{\mathbf{W}}^T \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} \tilde{\mathbf{W}} - 2 \mathbf{Y}^T \tilde{\mathbf{X}} \tilde{\mathbf{W}} + \mathbf{Y}^T \mathbf{Y} \right)
$$

Computing the Hessian matrix (the matrix of second-order partial derivatives) with respect to $\tilde{\mathbf{W}}$:

$$
\nabla_{\tilde{\mathbf{W}}}^2 J(\tilde{\mathbf{W}}) = \frac{1}{m} \tilde{\mathbf{X}}^T \tilde{\mathbf{X}}
$$

For any non-zero vector $\mathbf{v} \in \mathbb{R}^{n+1} \setminus \{\mathbf{0}\}$:

$$
\mathbf{v}^T \left( \nabla_{\tilde{\mathbf{W}}}^2 J \right) \mathbf{v} = \frac{1}{m} \mathbf{v}^T \tilde{\mathbf{X}}^T \tilde{\mathbf{X}} \mathbf{v} = \frac{1}{m} (\tilde{\mathbf{X}}\mathbf{v})^T (\tilde{\mathbf{X}}\mathbf{v}) = \frac{1}{m} \|\tilde{\mathbf{X}}\mathbf{v}\|_2^2 \ge 0
$$

Because the Euclidean norm squared $\|\tilde{\mathbf{X}}\mathbf{v}\|_2^2$ is non-negative for all $\mathbf{v}$, the Hessian matrix $\nabla_{\tilde{\mathbf{W}}}^2 J$ is positive semi-definite. Furthermore, because one reference region was omitted during preprocessing, $\tilde{\mathbf{X}}$ exhibits full column rank ($\operatorname{rank}(\tilde{\mathbf{X}}) = n + 1 = 9$). Consequently:

$$
\tilde{\mathbf{X}}\mathbf{v} \neq \mathbf{0} \quad \forall \mathbf{v} \neq \mathbf{0} \implies \mathbf{v}^T \left( \nabla_{\tilde{\mathbf{W}}}^2 J \right) \mathbf{v} > 0
$$

The Hessian is therefore **strictly positive definite**, proving that the cost surface $J(\mathbf{W}, b)$ is strictly convex. A strictly convex surface possesses a single unique global minimum and is free of local minima or saddle points, guaranteeing that gradient descent will converge to the global optimum under an appropriate learning rate.

---

## 4. Analytic Derivation of the Gradient Vectors

To minimize $J(\mathbf{W}, b)$ via gradient descent, we must compute the exact partial derivatives with respect to each parameter.

### 4.1 First-Principles Scalar Derivation

Consider the objective function expressed in scalar summation form:

$$
J(w_1, \dots, w_n, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \sum_{k=1}^{n} w_k x_k^{(i)} + b - y^{(i)} \right)^2
$$

Applying the chain rule of differential calculus to differentiate with respect to an arbitrary weight $w_j$:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{2m} \sum_{i=1}^{m} \frac{\partial}{\partial w_j} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

$$
\frac{\partial J}{\partial w_j} = \frac{1}{2m} \sum_{i=1}^{m} 2 \left( \hat{y}^{(i)} - y^{(i)} \right) \cdot \frac{\partial}{\partial w_j} \left( \sum_{k=1}^{n} w_k x_k^{(i)} + b - y^{(i)} \right)
$$

Because $\frac{\partial}{\partial w_j} (w_k x_k^{(i)}) = x_j^{(i)}$ when $k = j$ and $0$ otherwise:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$

Similarly, differentiating with respect to the scalar bias $b$:

$$
\frac{\partial J}{\partial b} = \frac{1}{2m} \sum_{i=1}^{m} 2 \left( \hat{y}^{(i)} - y^{(i)} \right) \cdot \frac{\partial}{\partial b} \left( \sum_{k=1}^{n} w_k x_k^{(i)} + b - y^{(i)} \right)
$$

Since $\frac{\partial}{\partial b}(b) = 1$:

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$

### 4.2 Vectorized Matrix Calculus Formulation

Expressing the scalar derivatives in vectorized form:

The collection of partial derivatives with respect to the weight vector forms the gradient vector $\nabla_{\mathbf{W}} J \in \mathbb{R}^{n \times 1}$:

$$
\nabla_{\mathbf{W}} J = \begin{bmatrix}
\frac{\partial J}{\partial w_1} \\
\frac{\partial J}{\partial w_2} \\
\vdots \\
\frac{\partial J}{\partial w_n}
\end{bmatrix} = \frac{1}{m} \begin{bmatrix}
\sum_{i=1}^m x_1^{(i)} e^{(i)} \\
\sum_{i=1}^m x_2^{(i)} e^{(i)} \\
\vdots \\
\sum_{i=1}^m x_n^{(i)} e^{(i)}
\end{bmatrix} = \frac{1}{m} \mathbf{X}^T \mathbf{E}
$$

where $\mathbf{E} = \hat{\mathbf{Y}} - \mathbf{Y} \in \mathbb{R}^{m \times 1}$. 

Dimensional verification:
- $\mathbf{X}^T$ has dimensions $(n \times m)$.
- $\mathbf{E}$ has dimensions $(m \times 1)$.
- Therefore, $\mathbf{X}^T \mathbf{E}$ has dimensions $(n \times 1)$, matching the shape of $\mathbf{W}$.

For the scalar intercept $b$, the gradient $\nabla_b J \in \mathbb{R}$ is the mean of the error vector:

$$
\nabla_b J = \frac{\partial J}{\partial b} = \frac{1}{m} \mathbf{1}_m^T \mathbf{E} = \frac{1}{m} \sum_{i=1}^{m} e^{(i)}
$$

### 4.3 Implementation of Cost and Gradient Functions

In `src/linear_regression.py`, these derivations are implemented in vectorized form:

```python
def compute_cost(X, Y, W, b):
    """
    Compute the half-MSE cost J(W, b) = (1 / 2m) * sum((Y_pred - Y)^2).
    """
    m = X.shape[0]
    Y_predict = predict(X, W, b)
    error = Y_predict - Y
    J = np.sum(error**2) / (2 * m)
    return float(J)

def compute_gradient(X, Y, W, b):
    """
    Compute the gradients dW and db analytically.
    dW = (1/m) * X.T @ (Y_pred - Y)
    db = (1/m) * sum(Y_pred - Y)
    """
    m = X.shape[0]
    Y_predict = predict(X, W, b)
    error = Y_predict - Y

    dW = (1 / m) * np.matmul(X.T, error)
    db = (1 / m) * np.sum(error)
    return dW, db
```

Vectorizing these computations replaces slow Python loops over samples with optimized BLAS operations in NumPy, yielding substantial performance gains.

---

## 5. Optimization Dynamics: Batch Gradient Descent

### 5.1 The Batch Gradient Descent Update Rule

Gradient descent is a first-order iterative optimization algorithm that updates parameters in the direction of steepest descent (the negative gradient):

$$
\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \alpha \nabla_{\mathbf{W}} J^{(t)} = \mathbf{W}^{(t)} - \frac{\alpha}{m} \mathbf{X}^T \mathbf{E}^{(t)}
$$

$$
b^{(t+1)} = b^{(t)} - \alpha \nabla_b J^{(t)} = b^{(t)} - \frac{\alpha}{m} \sum_{i=1}^{m} e^{(i)(t)}
$$

where:
- $t \in \{0, 1, 2, \dots, T-1\}$ denotes the iteration index.
- $\alpha \in \mathbb{R}^+$ denotes the learning rate (step size hyperparameter).
- The algorithm is termed **batch** gradient descent because every parameter update evaluates the sum of errors over the entire training set of $m = 935$ observations.

### 5.2 Algorithmic Pseudocode and Execution Flow

The training procedure is summarized in Algorithm 2.1:

```
Algorithm 2.1: First-Principles Batch Gradient Descent for Multiple Linear Regression
────────────────────────────────────────────────────────────────────────────────────────
Input:  Design matrix X in R^(m x n), Target vector Y in R^(m x 1),
        Learning rate alpha in R^+, Total iterations num_iters in Z^+
Output: Learned weights W in R^(n x 1), Learned bias b in R,
        Cost history J_history, Checkpoint iteration indices

1:  m, n <- shape(X)
2:  W <- zeros(n, 1)                      // Zero-weight initialization
3:  b <- 0.0                              // Zero-bias initialization
4:  J_history <- empty_list()
5:  iterations <- empty_list()
6:
7:  for t = 1 to num_iters do:
8:      // Compute analytical gradients across all m training examples
9:      dW, db <- compute_gradient(X, Y, W, b)
10:
11:     // Update parameters along the negative gradient vector
12:     W <- W - alpha * dW
13:     b <- b - alpha * db
14:
15:     // Checkpoint sampling cadence every 10 iterations
16:     if (t mod 10 == 0) then:
17:         J <- compute_cost(X, Y, W, b)
18:         append(J_history, J)
19:         append(iterations, t)
20:     end if
21: end for
22:
23: return W, b, J_history, iterations
────────────────────────────────────────────────────────────────────────────────────────
```

### 5.3 Implementation in Python

In `src/linear_regression.py`, `gradient_descent()` implements this procedure:

```python
def gradient_descent(X, Y, alpha, num_iters=1000):
    """
    Train multiple linear regression using batch gradient descent.
    """
    m, n_features = X.shape
    W, b = initialize_parameters(n_features)
    J_history = []
    iterations = []

    for i in range(1, num_iters + 1):
        dW, db = compute_gradient(X, Y, W, b)

        W = W - alpha * dW
        b = b - alpha * db

        if i % 10 == 0:
            J = compute_cost(X, Y, W, b)
            J_history.append(J)
            iterations.append(i)

    return W, b, J_history, iterations
```

With `num_iters=1000`, the algorithm executes 1,000 updates and records 100 checkpoints (iterations $10, 20, \dots, 1000$). Checkpointing every 10 steps tracks convergence trajectory while reducing logging overhead.

---

## 6. Algorithmic Alternatives: Gradient Descent versus the Normal Equation

In classical econometrics, linear regression parameters are frequently estimated via the closed-form **Normal Equation**:

$$
\tilde{\mathbf{W}}^* = (\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})^{-1} \tilde{\mathbf{X}}^T \mathbf{Y}
$$

Table 2.2 compares the theoretical properties of Batch Gradient Descent with the analytic Normal Equation.

**Table 2.2: Theoretical Comparison: Batch Gradient Descent vs. Normal Equation**

| Dimension | Batch Gradient Descent (Implemented) | Closed-Form Normal Equation |
| :--- | :--- | :--- |
| **Computational Complexity** | $\mathcal{O}(T \cdot m \cdot n)$ where $T = 1,000, m = 935, n = 8$ | $\mathcal{O}(n^3 + n^2 m)$ dominated by matrix inversion |
| **Memory Footprint** | $\mathcal{O}(m \cdot n)$ storing $\mathbf{X}$ and gradient updates | $\mathcal{O}(n^2)$ storing the Gram matrix $\tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$ |
| **Hyperparameter Requirements** | Requires tuning step size $\alpha$ and epochs $T$ | Analytical solution; zero hyperparameters |
| **Feature Scaling Dependency** | Highly sensitive; requires standardization to condition contours | Invariant to non-singular linear feature scaling |
| **Scalability to Big Data ($n \gg 10^4$)** | Scales effectively; readily parallelizable on GPUs | Fails when $n$ is large due to cubic $\mathcal{O}(n^3)$ inversion cost |
| **Singularity Vulnerability** | Stable even if $\tilde{\mathbf{X}}$ is rank-deficient | Fails if $\det(\tilde{\mathbf{X}}^T \tilde{\mathbf{X}}) = 0$ without pseudo-inverse |

Because our primary educational and architectural objective is implementing iterative gradient-based optimization from first principles, batch gradient descent was selected as the core engine.

---

## 7. Empirical Results: Learned Parameters and Actuarial Semantics

The model was trained on the training partition ($N_{\text{train}} = 935$) using the optimal learning rate $\alpha = 0.3$ for $T = 1,000$ iterations (the hyperparameter search is documented in Chapter 3). The optimization converged to the parameter values presented in Table 2.3.

**Table 2.3: Empirical Learned Weights and Actuarial Interpretation ($\alpha = 0.3, T = 1,000$)**

| Parameter | Associated Covariate | Variable Type | Learned Value | Actuarial & Econometric Interpretation |
| :--- | :--- | :--- | :---: | :--- |
| **$b$ (Bias)** | Constant Intercept | Baseline Intercept | **$+8,905.4513$** | Expected baseline annual medical charges (\$) for a female non-smoker residing in the Northeast with average age ($39.01$), average BMI ($30.62$), and average children ($1.08$). |
| **$w_1$** | `age` | Standardized Cont. | **$+3,515.4201$** | A one-standard-deviation increase in age ($13.99$ years) increases expected charges by $\$3,515.42$, equivalent to an annual age premium of **$+\$251.25$ per year of age**. |
| **$w_2$** | `sex` | Binary Indicator | **$-111.9070$** | Holding all physiological and geographic variables fixed, male beneficiaries incur $\$111.91$ lower annual charges than females. The small magnitude indicates negligible sex differences when controlling for other factors. |
| **$w_3$** | `bmi` | Standardized Cont. | **$+1,977.1322$** | A one-standard-deviation increase in BMI ($6.02\text{ kg/m}^2$) increases expected charges by $\$1,977.13$, equivalent to **$+\$328.38$ per $\text{kg/m}^2$ of BMI**. |
| **$w_4$** | `children` | Standardized Cont. | **$+625.1713$** | A one-standard-deviation increase in dependent children ($1.20$) increases expected charges by $\$625.17$, equivalent to **$+\$522.15$ per covered child**. |
| **$w_5$** | `smoker` | Binary Indicator | **$+22,874.4465$** | Holding all other covariates constant, active tobacco smokers incur an estimated **$+\$22,874.45$ surcharge** relative to non-smokers, representing the dominant linear cost driver. |
| **$w_6$** | `region_northwest` | Dummy Indicator | **$-465.7452$** | Living in the Northwest is associated with a $\$465.75$ lower expected charge relative to the baseline Northeast region. |
| **$w_7$** | `region_southeast` | Dummy Indicator | **$-936.0979$** | Living in the Southeast is associated with a $\$936.10$ lower expected charge relative to the Northeast, after controlling for higher regional BMI and smoking rates. |
| **$w_8$** | `region_southwest` | Dummy Indicator | **$-765.5825$** | Living in the Southwest is associated with a $\$765.58$ lower expected charge relative to the baseline Northeast region. |

### 7.1 Key Actuarial Findings

1. **Tobacco Consumption**: The coefficient $w_{\text{smoker}} = +22,874.45$ confirms smoking as the primary determinant of insurance charges in this dataset.
2. **De-Confounding Regional Averages**: In raw descriptive statistics (Chapter 1), the Southeast exhibited the highest unconditional mean charges ($\$14,735.41$). However, the multivariable regression coefficient for the Southeast is negative ($w_{\text{southeast}} = -936.10$). This demonstrates multivariate de-confounding: the higher raw regional charges in the Southeast are explained by higher average BMI and smoking prevalence rather than elevated regional medical costs per se.
3. **Biological Sex Attenuation**: The raw female-male charge difference of $+\$1,405.41$ shrinks to $-\$111.91$ after conditioning on smoking, BMI, and age, confirming that biological sex has little independent association with medical costs in this dataset.

---

## 8. Theoretical Assumptions and Econometric Diagnostic Boundaries

Multiple linear regression provides an interpretable baseline, but relies on several core theoretical assumptions (the classical Gauss-Markov conditions):

1. **Linearity in Parameters**: The conditional expectation $\mathbb{E}[Y \mid \mathbf{X}]$ is assumed to be an affine combination of the parameters:
   
   $$\mathbb{E}[Y \mid \mathbf{X}] = \mathbf{X}\mathbf{W} + b$$
   
   *Limitation*: Additive linear models cannot represent multiplicative interaction effects (such as the observed $\text{BMI} \times \text{smoker}$ interaction) without explicit interaction terms.
2. **Strict Exogeneity**: The error terms are assumed to have conditional expectation zero given the design matrix:
   
   $$\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{X}] = \mathbf{0}$$
   
   *Limitation*: Omitted confounders (e.g., specific clinical diagnoses, genetic risk factors, insurance plan benefit design) can introduce bias into the learned coefficients.
3. **Homoscedasticity**: The conditional variance of the disturbances is assumed constant across all covariate profiles:
   
   $$\operatorname{Var}(\boldsymbol{\epsilon} \mid \mathbf{X}) = \sigma^2 \mathbf{I}_m$$
   
   *Limitation*: High-charge and smoking cohorts display wider residual dispersion than low-charge non-smokers, violating homoscedasticity.
4. **Absence of Perfect Multicollinearity**: The design matrix must have full column rank ($\operatorname{rank}(\mathbf{X}) = n$). This condition was satisfied by omitting the Northeast region indicator.
5. **Observational vs. Causal Interpretation**: These regression coefficients represent conditional statistical associations rather than causal effects.

---

## 9. Modular Software Architecture

The regression framework was structured across two dedicated modules in `src/`:

**Table 2.4: Module Responsibilities and Architectural Mapping**

| Module File | Function Name | Input Arguments | Return Signatures | Primary Mathematical Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| `src/linear_regression.py` | `initialize_parameters()` | `n_features: int` | `(W, b)` | Allocates $\mathbf{W} = \mathbf{0}_{n \times 1}$ and $b = 0.0$. |
| `src/linear_regression.py` | `predict()` | `X, W, b` | `Y_pred: ndarray` | Computes forward affine pass $\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + b$. |
| `src/linear_regression.py` | `compute_cost()` | `X, Y, W, b` | `J: float` | Computes half-MSE objective $J = \frac{1}{2m} \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$. |
| `src/linear_regression.py` | `compute_gradient()` | `X, Y, W, b` | `(dW, db)` | Computes analytical gradients $\frac{1}{m}\mathbf{X}^T\mathbf{E}$ and $\frac{1}{m}\mathbf{1}^T\mathbf{E}$. |
| `src/linear_regression.py` | `gradient_descent()` | `X, Y, alpha, num_iters` | `(W, b, J_hist, iters)` | Executes batch gradient updates and logs cost checkpoints. |
| `src/metrics_calculation.py` | `mean_absolute_error()` | `Y, Y_pred` | `mae: float` | Computes linear error $\frac{1}{m} \sum \|y^{(i)} - \hat{y}^{(i)}\|$. |
| `src/metrics_calculation.py` | `root_mean_squared_error()` | `Y, Y_pred` | `rmse: float` | Computes quadratic error $\sqrt{\frac{1}{m} \sum (y^{(i)} - \hat{y}^{(i)})^2}$. |
| `src/metrics_calculation.py` | `r_squared()` | `Y, Y_pred` | `r2: float` | Computes explained variance $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$. |

---

## 10. Chapter Summary and Key Conclusions

This chapter established the theoretical, algebraic, and computational foundation of the multiple linear regression system:
1. **Mathematical Rigor**: Formalized the affine linear hypothesis in vectorized matrix notation ($\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + b\mathbf{1}_m$).
2. **Convex Optimization**: Formulated the Half-MSE loss function $J(\mathbf{W}, b)$, proved strict positive-definiteness of its Hessian, and established strict convexity.
3. **Analytic Gradient Derivation**: Derived the exact gradient equations $\nabla_{\mathbf{W}} J = \frac{1}{m}\mathbf{X}^T\mathbf{E}$ and $\nabla_b J = \frac{1}{m}\mathbf{1}_m^T\mathbf{E}$ from first principles.
4. **Vectorized Implementation**: Replaced sample loops with matrix operations using NumPy.
5. **Actuarial Interpretability**: Demonstrated that the trained weights quantify direct feature associations, identifying tobacco use ($+\$22,874.45$) as the dominant cost driver and resolving regional confounding.

Chapter 3 explores the dynamics of batch gradient descent through systematic learning-rate experiments and convergence diagnostics.
