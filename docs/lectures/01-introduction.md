# Lecture 1: Introduction to Machine Learning

## What is Machine Learning?

Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of following predefined rules, ML algorithms identify patterns and make decisions based on experience.

!!! note "Definition"
    **Machine Learning** is the field of study that gives computers the ability to learn without being explicitly programmed.

    *— Arthur Samuel, 1959*

## Types of Machine Learning

### 1. Supervised Learning

In supervised learning, we train models on labeled data where both inputs $X$ and outputs $Y$ are known.

**Mathematical Formulation:**

Given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}$, we want to learn a function $f: X \rightarrow Y$ such that:

$$
f(x_i) \approx y_i \quad \text{for all } i
$$

!!! example "Common Supervised Learning Tasks"
    - **Classification**: Predicting discrete labels (e.g., spam/not spam)
    - **Regression**: Predicting continuous values (e.g., house prices)

### 2. Unsupervised Learning

In unsupervised learning, we only have input data $X$ without corresponding labels. The goal is to discover hidden patterns or structure.

**Key Algorithms:**

- Clustering (e.g., K-means)
- Dimensionality Reduction (e.g., PCA)
- Anomaly Detection

### 3. Reinforcement Learning

An agent learns to make decisions by interacting with an environment and receiving rewards or penalties.

## The Learning Process

The typical machine learning workflow consists of several stages:

```python
# Pseudocode for a typical ML pipeline
def ml_pipeline(data):
    # 1. Data Preprocessing
    X_train, X_test, y_train, y_test = preprocess_and_split(data)

    # 2. Feature Engineering
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # 3. Model Training
    model = train_model(X_train_features, y_train)

    # 4. Evaluation
    predictions = model.predict(X_test_features)
    accuracy = evaluate(predictions, y_test)

    # 5. Optimization
    if accuracy < threshold:
        model = tune_hyperparameters(model, X_train_features, y_train)

    return model
```

!!! warning "Overfitting"
    Be careful not to overfit your model to the training data! A model that performs perfectly on training data but poorly on test data has failed to generalize.

## Loss Functions

A **loss function** (or cost function) measures how well our model's predictions match the actual values.

### Mean Squared Error (MSE)

For regression tasks, we often use MSE:

$$
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i; \theta))^2
$$

where:

- $\theta$ represents the model parameters
- $y_i$ is the true value
- $f(x_i; \theta)$ is the predicted value

### Cross-Entropy Loss

For classification tasks, we typically use cross-entropy:

$$
\mathcal{L}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
$$

where $C$ is the number of classes and $p_{i,c}$ is the predicted probability for class $c$.

## Gradient Descent

To minimize the loss function, we use **gradient descent**:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

where $\eta$ is the learning rate.

```python
import numpy as np

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    """
    Perform gradient descent to minimize the loss function.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : numpy.ndarray
        Target vector of shape (n_samples,)
    theta : numpy.ndarray
        Initial parameters of shape (n_features,)
    learning_rate : float
        Step size for gradient descent
    iterations : int
        Number of iterations to perform

    Returns:
    --------
    theta : numpy.ndarray
        Optimized parameters
    loss_history : list
        History of loss values
    """
    m = len(y)
    loss_history = []

    for i in range(iterations):
        # Compute predictions
        predictions = X.dot(theta)

        # Compute errors
        errors = predictions - y

        # Compute gradient
        gradient = (1/m) * X.T.dot(errors)

        # Update parameters
        theta = theta - learning_rate * gradient

        # Compute and store loss
        loss = (1/(2*m)) * np.sum(errors**2)
        loss_history.append(loss)

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")

    return theta, loss_history
```

!!! tip "Choosing Learning Rate"
    The learning rate $\eta$ is crucial:

    - **Too small**: Training will be very slow
    - **Too large**: Training may not converge or may diverge

    A good practice is to start with $\eta = 0.01$ and adjust based on performance.

## Bias-Variance Tradeoff

One of the fundamental concepts in machine learning is the **bias-variance tradeoff**.

- **Bias**: Error from overly simplistic assumptions (underfitting)
- **Variance**: Error from too much complexity (overfitting)

The total error can be decomposed as:

$$
\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

![Bias-Variance Tradeoff](../assets/images/bias-variance.png)

!!! important "Finding the Sweet Spot"
    The goal is to find a model complexity that minimizes the total error — neither too simple (high bias) nor too complex (high variance).

## Model Evaluation

### Training, Validation, and Test Sets

We typically split our data into three sets:

1. **Training Set (60-80%)**: Used to train the model
2. **Validation Set (10-20%)**: Used to tune hyperparameters
3. **Test Set (10-20%)**: Used for final evaluation

```python
from sklearn.model_selection import train_test_split

# Split data into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split train+val into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)  # 0.25 * 0.8 = 0.2 of total data

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")
```

### Cross-Validation

**K-fold cross-validation** provides a more robust estimate of model performance:

1. Split data into $K$ folds
2. For each fold $k$:
   - Train on $K-1$ folds
   - Validate on fold $k$
3. Average the $K$ validation scores

$$
\text{CV Score} = \frac{1}{K} \sum_{k=1}^{K} \text{Score}_k
$$

!!! note "Common K Values"
    - $K=5$ or $K=10$ are commonly used
    - Leave-One-Out CV: $K=n$ (number of samples)

## Summary

In this lecture, we covered:

- ✓ Types of machine learning (supervised, unsupervised, reinforcement)
- ✓ The learning process and ML pipeline
- ✓ Loss functions and optimization with gradient descent
- ✓ Bias-variance tradeoff
- ✓ Model evaluation techniques

!!! success "Next Steps"
    In [Lecture 2](02-linear-regression.md), we'll dive deep into linear regression, our first machine learning algorithm!

## Further Reading

- **Book**: "Pattern Recognition and Machine Learning" by Christopher Bishop
- **Paper**: "A Few Useful Things to Know About Machine Learning" by Pedro Domingos
- **Online**: [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
