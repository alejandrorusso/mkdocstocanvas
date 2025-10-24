# Lecture 2: Linear Regression

## Introduction

Linear regression is one of the simplest yet most powerful machine learning algorithms. It models the relationship between a dependent variable and one or more independent variables using a linear function.

!!! note "Learning Objectives"
    By the end of this lecture, you will understand:

    - The mathematical foundation of linear regression
    - How to derive the optimal solution
    - Gradient descent for linear regression
    - Regularization techniques (Ridge and Lasso)

## Simple Linear Regression

### The Model

In simple linear regression, we model the relationship between a single feature $x$ and a target $y$:

$$
y = \theta_0 + \theta_1 x + \epsilon
$$

where:

- $\theta_0$ is the **intercept** (bias term)
- $\theta_1$ is the **slope** (weight)
- $\epsilon$ is the random error term

### Vectorized Form

For multiple features, we write this in vector form:

$$
y = \theta^T x = \sum_{j=0}^{d} \theta_j x_j
$$

where $x_0 = 1$ (to account for the intercept).

!!! tip "Matrix Notation"
    For $n$ samples and $d$ features, we can write:

    $$
    \mathbf{y} = X\theta
    $$

    where $X \in \mathbb{R}^{n \times (d+1)}$ is the design matrix and $\theta \in \mathbb{R}^{d+1}$ is the parameter vector.

## Loss Function

We use **Mean Squared Error (MSE)** as our loss function:

$$
\mathcal{L}(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \theta^T x_i)^2 = \frac{1}{2n} \|y - X\theta\|^2
$$

!!! warning "Why the factor of 1/2?"
    The factor of $\frac{1}{2}$ is included for mathematical convenience — it cancels with the 2 from the derivative of the squared term.

## Analytical Solution: Normal Equation

We can find the optimal $\theta$ by setting the gradient to zero:

$$
\nabla_\theta \mathcal{L}(\theta) = 0
$$

This gives us the **normal equation**:

$$
\theta^* = (X^T X)^{-1} X^T y
$$

### Derivation

Starting with the loss function:

$$
\mathcal{L}(\theta) = \frac{1}{2n} (y - X\theta)^T(y - X\theta)
$$

Expanding:

$$
\mathcal{L}(\theta) = \frac{1}{2n} (y^T y - y^T X\theta - \theta^T X^T y + \theta^T X^T X\theta)
$$

Taking the gradient with respect to $\theta$:

$$
\nabla_\theta \mathcal{L}(\theta) = \frac{1}{n}(-X^T y + X^T X\theta)
$$

Setting to zero and solving:

$$
X^T X\theta = X^T y \implies \theta = (X^T X)^{-1} X^T y
$$

### Implementation

```python
import numpy as np

def normal_equation(X, y):
    """
    Compute the closed-form solution for linear regression.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : numpy.ndarray
        Target vector of shape (n_samples,)

    Returns:
    --------
    theta : numpy.ndarray
        Optimal parameters of shape (n_features,)
    """
    # Add intercept term
    X_with_intercept = np.c_[np.ones(X.shape[0]), X]

    # Compute (X^T X)^{-1} X^T y
    theta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    return theta

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

theta = normal_equation(X, y)
print(f"Intercept: {theta[0]:.4f}")
print(f"Slope: {theta[1]:.4f}")
```

!!! important "Computational Complexity"
    The normal equation has a time complexity of $O(d^3)$ due to matrix inversion, where $d$ is the number of features.

    For large $d$ (e.g., $d > 10,000$), gradient descent is often more efficient!

## Gradient Descent for Linear Regression

### Algorithm

For linear regression, the gradient is:

$$
\nabla_\theta \mathcal{L}(\theta) = \frac{1}{n} X^T (X\theta - y)
$$

The update rule becomes:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{n} X^T (X\theta_t - y)
$$

### Variants of Gradient Descent

#### 1. Batch Gradient Descent

Uses all samples in each iteration:

```python
def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """Batch gradient descent for linear regression."""
    n, d = X.shape
    X = np.c_[np.ones(n), X]  # Add intercept
    theta = np.zeros(d + 1)
    loss_history = []

    for i in range(iterations):
        # Compute gradient using all samples
        gradient = (1/n) * X.T @ (X @ theta - y)

        # Update parameters
        theta -= learning_rate * gradient

        # Track loss
        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        loss_history.append(loss)

    return theta, loss_history
```

#### 2. Stochastic Gradient Descent (SGD)

Updates parameters using one sample at a time:

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """Stochastic gradient descent for linear regression."""
    n, d = X.shape
    X = np.c_[np.ones(n), X]  # Add intercept
    theta = np.zeros(d + 1)
    loss_history = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)

        for i in indices:
            # Compute gradient using single sample
            gradient = (X[i] @ theta - y[i]) * X[i]

            # Update parameters
            theta -= learning_rate * gradient

        # Track loss after each epoch
        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        loss_history.append(loss)

    return theta, loss_history
```

#### 3. Mini-Batch Gradient Descent

A compromise between batch and stochastic:

```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    """Mini-batch gradient descent for linear regression."""
    n, d = X.shape
    X = np.c_[np.ones(n), X]  # Add intercept
    theta = np.zeros(d + 1)
    loss_history = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process in mini-batches
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_n = len(X_batch)

            # Compute gradient on mini-batch
            gradient = (1/batch_n) * X_batch.T @ (X_batch @ theta - y_batch)

            # Update parameters
            theta -= learning_rate * gradient

        # Track loss after each epoch
        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        loss_history.append(loss)

    return theta, loss_history
```

!!! tip "Choosing the Right Variant"
    - **Batch GD**: Best for small datasets, stable convergence
    - **SGD**: Fast updates, good for large datasets, noisy convergence
    - **Mini-batch GD**: Best of both worlds, most commonly used (batch size 32-256)

## Regularization

Regularization prevents overfitting by penalizing large parameter values.

### Ridge Regression (L2 Regularization)

Adds a penalty proportional to the square of the parameters:

$$
\mathcal{L}_{\text{Ridge}}(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \theta^T x_i)^2 + \frac{\lambda}{2} \sum_{j=1}^{d} \theta_j^2
$$

The closed-form solution becomes:

$$
\theta^* = (X^T X + \lambda I)^{-1} X^T y
$$

where $\lambda > 0$ is the regularization parameter.

```python
def ridge_regression(X, y, lambda_reg=1.0):
    """
    Ridge regression with L2 regularization.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    lambda_reg : float
        Regularization parameter

    Returns:
    --------
    theta : numpy.ndarray
        Regularized parameters
    """
    n, d = X.shape
    X = np.c_[np.ones(n), X]  # Add intercept

    # Create regularization matrix (don't penalize intercept)
    L = lambda_reg * np.eye(d + 1)
    L[0, 0] = 0  # Don't regularize intercept

    # Compute regularized solution
    theta = np.linalg.inv(X.T @ X + L) @ X.T @ y

    return theta
```

!!! note "Effect of Lambda"
    - $\lambda = 0$: No regularization (standard linear regression)
    - $\lambda \to \infty$: All parameters shrink towards zero
    - Larger $\lambda$ reduces model complexity

### Lasso Regression (L1 Regularization)

Uses absolute value penalty:

$$
\mathcal{L}_{\text{Lasso}}(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \theta^T x_i)^2 + \lambda \sum_{j=1}^{d} |\theta_j|
$$

!!! important "Feature Selection"
    Lasso can drive some parameters exactly to zero, effectively performing **feature selection**!

### Comparison

| Method | Penalty | Solution | Feature Selection |
|--------|---------|----------|-------------------|
| Ridge  | $\lambda \sum \theta_j^2$ | Closed-form | No |
| Lasso  | $\lambda \sum \|\theta_j\|$ | Iterative | Yes |

![Ridge vs Lasso](../assets/images/ridge-vs-lasso.png)

## Model Evaluation Metrics

### R² Score (Coefficient of Determination)

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

where $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ is the mean of $y$.

- $R^2 = 1$: Perfect fit
- $R^2 = 0$: Model performs as well as predicting the mean
- $R^2 < 0$: Model performs worse than predicting the mean

```python
def r2_score(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)
```

### Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

RMSE is in the same units as $y$, making it interpretable.

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

MAE is less sensitive to outliers than MSE/RMSE.

!!! warning "Be Careful with Metrics"
    Different metrics have different properties:

    - **MSE/RMSE**: Penalizes large errors more heavily
    - **MAE**: Treats all errors equally
    - **R²**: Scale-invariant, but can be misleading with non-linear relationships

## Complete Example

Here's a full example bringing everything together:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train using normal equation
theta_normal = normal_equation(X_train, y_train)

# Train using gradient descent
theta_gd, loss_history = batch_gradient_descent(
    X_train, y_train, learning_rate=0.01, iterations=1000
)

# Make predictions
X_train_with_intercept = np.c_[np.ones(len(X_train)), X_train]
X_test_with_intercept = np.c_[np.ones(len(X_test)), X_test]

y_pred_train = X_train_with_intercept @ theta_gd
y_pred_test = X_test_with_intercept @ theta_gd

# Evaluate
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Plot results
plt.figure(figsize=(12, 4))

# Plot 1: Data and fitted line
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test data')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_line_with_intercept = np.c_[np.ones(len(X_line)), X_line]
y_line = X_line_with_intercept @ theta_gd
plt.plot(X_line, y_line, 'r-', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Fit')

# Plot 2: Loss history
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.grid(True)

plt.tight_layout()
plt.savefig('linear_regression_results.png')
plt.show()
```

## Summary

In this lecture, we covered:

- ✓ Linear regression model and its mathematical formulation
- ✓ Analytical solution using the normal equation
- ✓ Iterative optimization using gradient descent (batch, SGD, mini-batch)
- ✓ Regularization techniques (Ridge and Lasso)
- ✓ Evaluation metrics (R², RMSE, MAE)

!!! success "What's Next?"
    In [Lecture 3](03-neural-networks.md), we'll build on these foundations to explore neural networks!

    Also, check out [Lab 2](../labs/lab2.md) to implement linear regression from scratch!
