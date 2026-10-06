# Lab 1: Python Basics for Machine Learning

**Due Date**: Week 3
**Points**: 100
**Submission**: Submit your `.py` file and a PDF report

## Overview

This lab will familiarize you with essential Python libraries and tools used in machine learning: NumPy, Matplotlib, and basic data manipulation.

!!! important "Learning Objectives"
    By completing this lab, you will:

    - Master NumPy array operations
    - Create visualizations with Matplotlib
    - Implement basic statistical computations
    - Practice vectorized operations for efficiency

## Setup

First, ensure you have the required packages installed:

```bash
pip install numpy matplotlib scipy
```

!!! note "Environment"
    We recommend using Python 3.8 or later. You can use Jupyter notebooks, Google Colab, or plain Python scripts.

## Part 1: NumPy Fundamentals (30 points)

### Task 1.1: Array Operations (10 points)

Create a Python function that performs the following operations:

```python
import numpy as np

def array_operations():
    """
    Perform basic NumPy array operations.

    Returns:
    --------
    results : dict
        Dictionary containing results of various operations
    """
    # TODO: Create a 1D array of integers from 0 to 99
    arr = None

    # TODO: Reshape it to a 10x10 matrix
    matrix = None

    # TODO: Compute the mean of each row
    row_means = None

    # TODO: Compute the standard deviation of each column
    col_stds = None

    # TODO: Find the indices of the maximum value in the original array
    max_idx = None

    return {
        'array': arr,
        'matrix': matrix,
        'row_means': row_means,
        'col_stds': col_stds,
        'max_idx': max_idx
    }
```

!!! tip "Hint"
    Use `np.arange()`, `np.reshape()`, `np.mean()`, `np.std()`, and `np.argmax()`.

### Task 1.2: Matrix Operations (10 points)

Implement matrix operations without using loops:

```python
def matrix_operations(A, B):
    """
    Perform matrix operations.

    Parameters:
    -----------
    A : numpy.ndarray
        Matrix of shape (m, n)
    B : numpy.ndarray
        Matrix of shape (n, p)

    Returns:
    --------
    results : dict
        Dictionary containing results
    """
    # TODO: Matrix multiplication C = AB
    C = None

    # TODO: Element-wise multiplication
    elementwise = None  # Note: B must be broadcastable to A's shape

    # TODO: Transpose of A
    A_T = None

    # TODO: Frobenius norm of A: sqrt(sum of squared elements)
    frobenius_norm = None

    return {
        'matrix_product': C,
        'elementwise_product': elementwise,
        'transpose': A_T,
        'frobenius_norm': frobenius_norm
    }
```

### Task 1.3: Broadcasting (10 points)

Demonstrate understanding of NumPy broadcasting:

```python
def broadcasting_example():
    """
    Demonstrate NumPy broadcasting.

    Returns:
    --------
    normalized : numpy.ndarray
        Normalized matrix
    """
    # Create a 5x3 matrix of random numbers
    X = np.random.randn(5, 3)

    # TODO: Normalize each column to have mean 0 and std 1
    # Hint: Compute column means and stds, then use broadcasting
    mean = None
    std = None
    normalized = None

    # Verify: each column should have mean ≈ 0 and std ≈ 1
    print("Column means:", np.mean(normalized, axis=0))
    print("Column stds:", np.std(normalized, axis=0))

    return normalized
```

!!! warning "Common Mistake"
    Remember that `axis=0` operates along rows (computing column statistics), while `axis=1` operates along columns (computing row statistics).

## Part 2: Data Visualization (30 points)

### Task 2.1: Basic Plotting (10 points)

Create visualizations of mathematical functions:

```python
import matplotlib.pyplot as plt

def plot_functions():
    """
    Plot various activation functions used in machine learning.
    """
    x = np.linspace(-5, 5, 200)

    # TODO: Compute sigmoid function
    sigmoid = 1 / (1 + np.exp(-x))

    # TODO: Compute tanh function
    tanh = np.tanh(x)

    # TODO: Compute ReLU function
    relu = np.maximum(0, x)

    # Create plot with all three functions
    plt.figure(figsize=(10, 6))

    # TODO: Plot sigmoid
    plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)

    # TODO: Plot tanh
    plt.plot(x, tanh, label='Tanh', linewidth=2)

    # TODO: Plot ReLU
    plt.plot(x, relu, label='ReLU', linewidth=2)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Common Activation Functions', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Task 2.2: Scatter Plots and Data Distribution (10 points)

Visualize data distributions:

```python
def visualize_data_distribution():
    """
    Create scatter plots and histograms to understand data distribution.
    """
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)

    # TODO: Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # TODO: Plot 1 - Scatter plot colored by class
    # axes[0, 0].scatter(...)

    # TODO: Plot 2 - Histogram of first feature
    # axes[0, 1].hist(...)

    # TODO: Plot 3 - Histogram of second feature
    # axes[1, 0].hist(...)

    # TODO: Plot 4 - 2D histogram (density)
    # axes[1, 1].hist2d(...)

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Task 2.3: Visualizing Loss Curves (10 points)

Create a plot showing training vs. validation loss:

```python
def plot_loss_curves():
    """
    Plot training and validation loss curves.
    """
    # Simulate loss curves
    epochs = np.arange(1, 101)
    train_loss = 2.0 * np.exp(-0.05 * epochs) + 0.1
    val_loss = 2.0 * np.exp(-0.04 * epochs) + 0.2 + 0.1 * np.sin(epochs / 10)

    # TODO: Create plot with both curves
    plt.figure(figsize=(10, 6))

    # TODO: Plot training loss
    # plt.plot(...)

    # TODO: Plot validation loss
    # plt.plot(...)

    # Add labels and formatting
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # TODO: Mark the point of best validation loss
    best_epoch = np.argmin(val_loss)
    # plt.axvline(...)
    # plt.scatter(...)

    plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## Part 3: Statistical Operations (20 points)

### Task 3.1: Computing Statistics (10 points)

Implement functions to compute various statistics:

```python
def compute_statistics(data):
    """
    Compute various statistics for a dataset.

    Parameters:
    -----------
    data : numpy.ndarray
        1D array of numerical values

    Returns:
    --------
    stats : dict
        Dictionary of computed statistics
    """
    # TODO: Compute mean
    mean = None

    # TODO: Compute median
    median = None

    # TODO: Compute standard deviation
    std = None

    # TODO: Compute 25th and 75th percentiles
    q25 = None
    q75 = None

    # TODO: Compute interquartile range (IQR)
    iqr = None

    return {
        'mean': mean,
        'median': median,
        'std': std,
        'q25': q25,
        'q75': q75,
        'iqr': iqr
    }
```

### Task 3.2: Correlation Analysis (10 points)

Compute and visualize correlations:

```python
def correlation_analysis():
    """
    Analyze correlations between features.
    """
    # Generate correlated data
    np.random.seed(42)
    n = 100
    X1 = np.random.randn(n)
    X2 = 0.8 * X1 + 0.6 * np.random.randn(n)
    X3 = -0.5 * X1 + 0.866 * np.random.randn(n)

    # Create data matrix
    X = np.column_stack([X1, X2, X3])

    # TODO: Compute correlation matrix
    # Hint: Use np.corrcoef()
    corr_matrix = None

    # TODO: Create a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks([0, 1, 2], ['X1', 'X2', 'X3'])
    plt.yticks([0, 1, 2], ['X1', 'X2', 'X3'])
    plt.title('Feature Correlation Matrix')

    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha='center', va='center', color='black')

    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    return corr_matrix
```

## Part 4: Vectorization for Efficiency (20 points)

### Task 4.1: Timing Comparison (10 points)

Compare loop-based vs. vectorized implementations:

```python
import time

def euclidean_distance_loop(X, Y):
    """
    Compute pairwise Euclidean distances using loops.

    Parameters:
    -----------
    X : numpy.ndarray
        Matrix of shape (n, d)
    Y : numpy.ndarray
        Matrix of shape (m, d)

    Returns:
    --------
    distances : numpy.ndarray
        Distance matrix of shape (n, m)
    """
    n, d = X.shape
    m = Y.shape[0]
    distances = np.zeros((n, m))

    # TODO: Implement using nested loops
    for i in range(n):
        for j in range(m):
            distances[i, j] = np.sqrt(np.sum((X[i] - Y[j])**2))

    return distances

def euclidean_distance_vectorized(X, Y):
    """
    Compute pairwise Euclidean distances using vectorization.

    Parameters:
    -----------
    X : numpy.ndarray
        Matrix of shape (n, d)
    Y : numpy.ndarray
        Matrix of shape (m, d)

    Returns:
    --------
    distances : numpy.ndarray
        Distance matrix of shape (n, m)
    """
    # TODO: Implement using vectorized operations
    # Hint: Use broadcasting and the formula:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y

    X_sq = None  # Shape: (n, 1)
    Y_sq = None  # Shape: (1, m)
    XY = None    # Shape: (n, m)

    distances = None

    return distances

def compare_implementations():
    """Compare timing of loop vs vectorized implementations."""
    # Generate test data
    n, m, d = 100, 150, 50
    X = np.random.randn(n, d)
    Y = np.random.randn(m, d)

    # Time loop implementation
    start = time.time()
    dist_loop = euclidean_distance_loop(X, Y)
    time_loop = time.time() - start

    # Time vectorized implementation
    start = time.time()
    dist_vec = euclidean_distance_vectorized(X, Y)
    time_vec = time.time() - start

    # Verify results are the same
    print(f"Max difference: {np.max(np.abs(dist_loop - dist_vec)):.10f}")
    print(f"Loop time: {time_loop:.4f} seconds")
    print(f"Vectorized time: {time_vec:.4f} seconds")
    print(f"Speedup: {time_loop / time_vec:.2f}x")
```

!!! success "Expected Result"
    The vectorized implementation should be **10-100x faster** than the loop version!

### Task 4.2: Batch Processing (10 points)

Implement batch processing for a simple operation:

```python
def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def process_batch(X, W, b):
    """
    Process a batch of data through a linear layer with sigmoid.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data of shape (batch_size, input_dim)
    W : numpy.ndarray
        Weight matrix of shape (input_dim, output_dim)
    b : numpy.ndarray
        Bias vector of shape (output_dim,)

    Returns:
    --------
    output : numpy.ndarray
        Processed output of shape (batch_size, output_dim)
    """
    # TODO: Compute X @ W + b and apply sigmoid
    # All operations should be vectorized
    z = None
    output = None

    return output
```

## Submission Guidelines

!!! important "What to Submit"
    Submit **two files**:

    1. **`lab1.py`**: Your complete Python code with all implemented functions
    2. **`lab1_report.pdf`**: A PDF report including:
        - All generated plots
        - Answers to questions
        - Timing comparisons from Part 4
        - Brief discussion of what you learned

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Part 1: NumPy | 30 | Correct implementation of array operations |
| Part 2: Visualization | 30 | Clear, well-labeled plots |
| Part 3: Statistics | 20 | Accurate statistical computations |
| Part 4: Vectorization | 20 | Efficient vectorized code, speedup demonstrated |
| **Total** | **100** | |

!!! warning "Academic Integrity"
    This is an individual assignment. You may discuss concepts with classmates, but all code and writeup must be your own work.

## Tips for Success

- Test your functions with small examples first
- Use `assert` statements to verify correctness
- Read NumPy documentation when stuck
- Start early — don't wait until the deadline!

Good luck!
