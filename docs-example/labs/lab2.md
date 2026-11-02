# Lab 2: Implementing Linear Regression

**Due Date**: Week 6
**Points**: 100
**Submission**: Submit your `.py` file, notebook, and a PDF report

## Overview

In this lab, you will implement linear regression from scratch using only NumPy. You'll explore both the analytical solution (normal equation) and iterative optimization (gradient descent), and compare their performance on real-world datasets.

!!! important "Learning Objectives"
    - Implement linear regression without using scikit-learn
    - Understand the normal equation and gradient descent
    - Compare different optimization methods
    - Apply regularization techniques
    - Evaluate model performance

## Setup

```bash
pip install numpy matplotlib pandas scikit-learn
```

Download the housing dataset from the course website or use `sklearn.datasets.load_boston()`.

!!! note "Starter Code"
    A skeleton file `lab2_skeleton.py` is provided on Canvas. Start with this template.

## Part 1: Data Preprocessing (15 points)

### Task 1.1: Load and Explore Data (5 points)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load and split the California housing dataset.

    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Training and test sets
    feature_names : list
        Names of features
    """
    # TODO: Load the California housing dataset
    data = None

    # TODO: Extract features and targets
    X = None
    y = None
    feature_names = None

    # TODO: Split into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = None, None, None, None

    # Print dataset information
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Features: {feature_names}")

    return X_train, X_test, y_train, y_test, feature_names
```

### Task 1.2: Feature Normalization (10 points)

Implement feature normalization (standardization):

```python
def normalize_features(X_train, X_test):
    """
    Normalize features to have mean 0 and std 1.

    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Test features

    Returns:
    --------
    X_train_norm, X_test_norm : numpy.ndarray
        Normalized features
    mean, std : numpy.ndarray
        Mean and std used for normalization (for future use)
    """
    # TODO: Compute mean and std from training set
    mean = None
    std = None

    # TODO: Normalize both training and test sets using training statistics
    X_train_norm = None
    X_test_norm = None

    return X_train_norm, X_test_norm, mean, std
```

!!! warning "Important"
    Always compute normalization statistics from the **training set only**, then apply them to both training and test sets. Never use test set statistics!

## Part 2: Normal Equation (20 points)

### Task 2.1: Implement Normal Equation (15 points)

```python
class LinearRegressionNormalEq:
    """Linear Regression using the Normal Equation."""

    def __init__(self):
        self.theta = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fit the model using the normal equation.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : numpy.ndarray
            Target vector of shape (n_samples,)
        """
        # TODO: Add intercept term (column of ones)
        X_with_intercept = None

        # TODO: Compute theta using (X^T X)^{-1} X^T y
        self.theta = None

        # TODO: Separate intercept and weights
        self.intercept = None
        self.theta = None  # Remaining weights

    def predict(self, X):
        """
        Make predictions.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features)

        Returns:
        --------
        predictions : numpy.ndarray
            Predicted values of shape (n_samples,)
        """
        # TODO: Compute predictions
        predictions = None

        return predictions

    def score(self, X, y):
        """
        Compute R² score.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            True values

        Returns:
        --------
        r2 : float
            R² score
        """
        # TODO: Compute predictions
        y_pred = None

        # TODO: Compute R² = 1 - (SS_res / SS_tot)
        ss_res = None
        ss_tot = None
        r2 = None

        return r2
```

### Task 2.2: Test Normal Equation (5 points)

```python
def test_normal_equation():
    """Test the normal equation implementation."""
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_data()
    X_train_norm, X_test_norm, _, _ = normalize_features(X_train, X_test)

    # Train model
    model = LinearRegressionNormalEq()
    model.fit(X_train_norm, y_train)

    # Evaluate
    train_r2 = model.score(X_train_norm, y_train)
    test_r2 = model.score(X_test_norm, y_test)

    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    return model
```

## Part 3: Gradient Descent (30 points)

### Task 3.1: Batch Gradient Descent (15 points)

```python
class LinearRegressionGD:
    """Linear Regression using Gradient Descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.intercept = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit the model using batch gradient descent.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : numpy.ndarray
            Target vector of shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # TODO: Initialize parameters to zeros
        self.theta = None
        self.intercept = None

        # Gradient descent loop
        for i in range(self.n_iterations):
            # TODO: Compute predictions
            y_pred = None

            # TODO: Compute errors
            errors = None

            # TODO: Compute gradients
            d_theta = None
            d_intercept = None

            # TODO: Update parameters
            self.theta = None
            self.intercept = None

            # TODO: Compute and store loss (MSE)
            loss = None
            self.loss_history.append(loss)

            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict(self, X):
        """Make predictions."""
        return X @ self.theta + self.intercept

    def score(self, X, y):
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
```

### Task 3.2: Mini-Batch Gradient Descent (15 points)

```python
class LinearRegressionMiniBatchGD:
    """Linear Regression using Mini-Batch Gradient Descent."""

    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=32):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.theta = None
        self.intercept = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit the model using mini-batch gradient descent.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target vector
        """
        n_samples, n_features = X.shape

        # TODO: Initialize parameters
        self.theta = np.zeros(n_features)
        self.intercept = 0

        # Epoch loop
        for epoch in range(self.n_epochs):
            # TODO: Shuffle data
            indices = None
            X_shuffled = None
            y_shuffled = None

            # Mini-batch loop
            for i in range(0, n_samples, self.batch_size):
                # TODO: Get mini-batch
                X_batch = None
                y_batch = None

                # TODO: Compute predictions for batch
                y_pred_batch = None

                # TODO: Compute errors for batch
                errors_batch = None

                # TODO: Compute gradients
                batch_n = None
                d_theta = None
                d_intercept = None

                # TODO: Update parameters
                self.theta -= self.learning_rate * d_theta
                self.intercept -= self.learning_rate * d_intercept

            # Compute loss for entire dataset
            y_pred = self.predict(X)
            loss = np.mean((y - y_pred)**2)
            self.loss_history.append(loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        """Make predictions."""
        return X @ self.theta + self.intercept

    def score(self, X, y):
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
```

## Part 4: Regularization (20 points)

### Task 4.1: Ridge Regression (10 points)

Implement Ridge regression (L2 regularization):

$$
\mathcal{L}(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \theta^T x_i)^2 + \frac{\lambda}{2} \|\theta\|^2
$$

```python
class RidgeRegression:
    """Ridge Regression with L2 regularization."""

    def __init__(self, alpha=1.0):
        """
        Parameters:
        -----------
        alpha : float
            Regularization strength (λ)
        """
        self.alpha = alpha
        self.theta = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fit Ridge regression using closed-form solution.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target vector
        """
        n_samples, n_features = X.shape

        # Add intercept term
        X_with_intercept = np.c_[np.ones(n_samples), X]

        # TODO: Create regularization matrix (don't penalize intercept)
        L = None

        # TODO: Compute regularized solution: (X^T X + λI)^{-1} X^T y
        self.theta = None

        # TODO: Separate intercept and weights
        self.intercept = None
        self.theta = None

    def predict(self, X):
        """Make predictions."""
        return X @ self.theta + self.intercept

    def score(self, X, y):
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
```

### Task 4.2: Regularization Strength Analysis (10 points)

Analyze the effect of regularization strength:

```python
def analyze_regularization():
    """
    Analyze the effect of different regularization strengths.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_data()
    X_train_norm, X_test_norm, _, _ = normalize_features(X_train, X_test)

    # Test different alpha values
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores = []
    test_scores = []

    for alpha in alphas:
        # TODO: Train Ridge model with this alpha
        model = None
        # model.fit(...)

        # TODO: Compute scores
        train_score = None
        test_score = None

        train_scores.append(train_score)
        test_scores.append(test_score)

        print(f"Alpha = {alpha:.3f}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

    # TODO: Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_scores, 'o-', label='Training')
    plt.semilogx(alphas, test_scores, 's-', label='Test')
    plt.xlabel('Regularization Strength (α)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Effect of Regularization on Model Performance', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('regularization_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
```

!!! tip "Choosing Alpha"
    - Small α: Little regularization, risk of overfitting
    - Large α: Strong regularization, risk of underfitting
    - Use cross-validation to find optimal α

## Part 5: Comparison and Analysis (15 points)

### Task 5.1: Compare All Methods (10 points)

```python
def compare_methods():
    """
    Compare all implemented methods.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    X_train_norm, X_test_norm, _, _ = normalize_features(X_train, X_test)

    results = {}

    # Normal Equation
    print("Training Normal Equation...")
    model_ne = LinearRegressionNormalEq()
    # TODO: Train and evaluate

    # Batch Gradient Descent
    print("\nTraining Batch Gradient Descent...")
    model_gd = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    # TODO: Train and evaluate

    # Mini-Batch Gradient Descent
    print("\nTraining Mini-Batch Gradient Descent...")
    model_mb = LinearRegressionMiniBatchGD(
        learning_rate=0.01, n_epochs=100, batch_size=32
    )
    # TODO: Train and evaluate

    # Ridge Regression
    print("\nTraining Ridge Regression...")
    model_ridge = RidgeRegression(alpha=1.0)
    # TODO: Train and evaluate

    # TODO: Create comparison table
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Method':<30} {'Train R²':<12} {'Test R²':<12}")
    print("-"*60)
    # Print results...

    return results
```

### Task 5.2: Visualize Loss Curves (5 points)

```python
def plot_loss_curves(model_gd, model_mb):
    """
    Plot loss curves for gradient descent methods.

    Parameters:
    -----------
    model_gd : LinearRegressionGD
        Trained batch GD model
    model_mb : LinearRegressionMiniBatchGD
        Trained mini-batch GD model
    """
    # TODO: Create plot comparing loss curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    # TODO: Plot batch GD loss
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Batch Gradient Descent')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # TODO: Plot mini-batch GD loss
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Mini-Batch Gradient Descent')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## Bonus: Feature Importance (10 points extra credit)

Visualize which features are most important:

```python
def plot_feature_importance(model, feature_names):
    """
    Plot feature importance based on coefficient magnitudes.

    Parameters:
    -----------
    model : trained model
        A trained linear regression model
    feature_names : list
        Names of features
    """
    # TODO: Get absolute values of coefficients
    importance = None

    # TODO: Sort by importance
    indices = None
    sorted_features = None
    sorted_importance = None

    # TODO: Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_importance)), sorted_importance)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('|Coefficient|', fontsize=12)
    plt.title('Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## Submission Guidelines

!!! important "What to Submit"
    Submit **three files**:

    1. **`lab2.py`**: Complete implementation of all classes and functions
    2. **`lab2_analysis.ipynb`** (optional): Jupyter notebook with experiments
    3. **`lab2_report.pdf`**: Report including:
        - All generated plots
        - Comparison table of all methods
        - Discussion of results
        - Answers to analysis questions

## Analysis Questions

Include answers to these questions in your report:

1. **Convergence**: How many iterations did batch gradient descent need to converge? How does this compare to mini-batch GD?

2. **Learning Rate**: What happens if you use a learning rate that's too large or too small? Experiment and report your findings.

3. **Regularization**: What value of α gave the best test performance? Why?

4. **Comparison**: Which method would you use in practice? Consider accuracy, speed, and scalability.

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Part 1: Data Preprocessing | 15 | Correct loading and normalization |
| Part 2: Normal Equation | 20 | Correct implementation and evaluation |
| Part 3: Gradient Descent | 30 | Both variants working correctly |
| Part 4: Regularization | 20 | Ridge regression implementation |
| Part 5: Comparison | 15 | Thorough analysis and visualizations |
| Bonus: Feature Importance | 10 | Extra credit |
| **Total** | **100 (+ 10 bonus)** | |

!!! warning "Testing Your Code"
    Make sure to test your implementations:

    ```python
    # Quick sanity check
    X_simple = np.array([[1], [2], [3], [4]])
    y_simple = np.array([2, 4, 6, 8])

    model = LinearRegressionNormalEq()
    model.fit(X_simple, y_simple)
    print(f"Intercept: {model.intercept:.2f} (should be ≈0)")
    print(f"Slope: {model.theta[0]:.2f} (should be ≈2)")
    ```

## Tips for Success

- Start with simple test cases before using the full dataset
- Verify your gradient calculations are correct
- Use `assert` to check shapes and intermediate values
- Plot loss curves to debug convergence issues
- Compare your results with scikit-learn as a sanity check

Good luck with your implementation!
