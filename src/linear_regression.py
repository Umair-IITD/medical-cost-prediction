import numpy as np

def initialize_parameters(n_features):
    """Returns a weight of shape (n,1) and a scalar bias."""

    W = np.zeros((n_features, 1))
    b = 0.0

    return W, b

def predict(X, W, b):
    """Predicts the insurance charge for given input using weights and bias"""

    Y_predict = np.matmul(X, W) + b

    return Y_predict

def compute_cost(X, Y, W, b):
    """
        Compute the mean squared error cost with a 1/(2m) factor.
        
        X: (m, n_features)
        Y: (m, 1)
        W: (n_features, 1)
        b: scalar
        
        Returns: J (scalar cost)
    """

    m = X.shape[0]

    Y_predict = predict(X, W, b)
    error = Y_predict - Y

    squared_error = error**2

    J = np.sum(squared_error)/(2*m)

    return float(J)

def compute_gradient(X, Y, W, b):
    """
    Compute the gradient dJ/dW for a given iteration
    dJ/dW[j] = (1/m)*Sum((Y_predict[i] - Y_train[i])*X[i][j])
    dJ/db = (1/m)Sum(Y_predict[i] - Y_train[i])
    
    Returns: Gradient (scalar slopes)
    """

    m = X.shape[0]
    Y_predict = predict(X, W, b)
    error = Y_predict - Y

    dW = (1/m) * np.matmul(X.T, error)
    db = (1/m) * np.sum(error)

    return dW, db

def gradient_descent(X, Y, alpha, num_iters=1000):
    """
    Train multiple linear regression using batch gradient descent algorithm
    
    W = W - alpha*dW
    b = b - alpha*db
    
    Returns:
        W: trained weights, shape (n_features,1)
        b: trained scalar bias
        J_history: recorded costs
        iterations: iteration numbers for the recorded costs
    """

    m,n_features = X.shape

    W,b = initialize_parameters(n_features)
    J_history = []
    iterations = []

    for i in range(1, num_iters+1):
        dW, db = compute_gradient(X, Y, W, b)

        W = W - alpha*dW
        b = b - alpha*db

        if i%10 == 0:
            J = compute_cost(X, Y, W, b)
            J_history.append(J)
            iterations.append(i)

    return W, b, J_history, iterations

