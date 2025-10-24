# Lecture 3: Neural Networks

## Introduction

Neural networks are computational models inspired by biological neurons in the brain. They consist of interconnected layers of artificial neurons that can learn complex patterns from data.

!!! note "From Linear to Non-Linear"
    While linear regression can only model linear relationships, neural networks can approximate any continuous function thanks to their non-linear activation functions and layered structure.

## The Perceptron: A Single Neuron

### Mathematical Model

A single neuron (perceptron) computes:

$$
\hat{y} = f\left(\sum_{i=1}^{d} w_i x_i + b\right) = f(w^T x + b)
$$

where:

- $x \in \mathbb{R}^d$ is the input vector
- $w \in \mathbb{R}^d$ is the weight vector
- $b \in \mathbb{R}$ is the bias term
- $f$ is the activation function

![Perceptron Diagram](../assets/images/perceptron.png)

### Activation Functions

The activation function $f$ introduces non-linearity:

#### 1. Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Properties:

- Output range: $(0, 1)$
- Smooth and differentiable
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

```python
import numpy as np

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)
```

#### 2. Hyperbolic Tangent (tanh)

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

Properties:

- Output range: $(-1, 1)$
- Zero-centered (better than sigmoid)
- Derivative: $\tanh'(z) = 1 - \tanh^2(z)$

```python
def tanh(z):
    """Tanh activation function."""
    return np.tanh(z)

def tanh_derivative(z):
    """Derivative of tanh function."""
    return 1 - np.tanh(z)**2
```

#### 3. ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

Properties:

- Most popular activation for hidden layers
- Computationally efficient
- Helps avoid vanishing gradient problem
- Derivative: $\text{ReLU}'(z) = \mathbb{1}_{z > 0}$

```python
def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU function."""
    return (z > 0).astype(float)
```

!!! warning "Dying ReLU Problem"
    ReLU neurons can sometimes "die" during training if they get stuck outputting zero for all inputs. Solutions include:

    - Using Leaky ReLU: $\text{LeakyReLU}(z) = \max(0.01z, z)$
    - Proper weight initialization
    - Lower learning rates

## Multi-Layer Perceptron (MLP)

### Architecture

An MLP consists of:

1. **Input layer**: Receives the features $x$
2. **Hidden layers**: One or more layers that learn representations
3. **Output layer**: Produces the final prediction

For a network with one hidden layer:

$$
\begin{align}
z^{[1]} &= W^{[1]} x + b^{[1]} \\
a^{[1]} &= f^{[1]}(z^{[1]}) \\
z^{[2]} &= W^{[2]} a^{[1]} + b^{[2]} \\
\hat{y} &= f^{[2]}(z^{[2]})
\end{align}
$$

where:

- $W^{[l]}$ and $b^{[l]}$ are weights and biases for layer $l$
- $z^{[l]}$ is the pre-activation (weighted sum)
- $a^{[l]}$ is the activation (output after applying $f$)

### Forward Propagation

```python
class NeuralNetwork:
    """Simple 2-layer neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network parameters.

        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of neurons in hidden layer
        output_size : int
            Number of output neurons
        """
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """
        Forward propagation.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, input_size)

        Returns:
        --------
        output : numpy.ndarray
            Network output of shape (n_samples, output_size)
        cache : dict
            Intermediate values needed for backpropagation
        """
        # Hidden layer
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)

        # Output layer
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid(Z2)  # For binary classification

        # Store values for backpropagation
        cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

        return A2, cache
```

## Backpropagation

Backpropagation is the algorithm for computing gradients efficiently using the **chain rule**.

### The Chain Rule

For a composition of functions $f(g(x))$:

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
$$

### Derivation for a 2-Layer Network

Given loss $\mathcal{L}$, we compute:

**Output Layer Gradients:**

$$
\begin{align}
\delta^{[2]} &= \frac{\partial \mathcal{L}}{\partial z^{[2]}} = \frac{\partial \mathcal{L}}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \\
\frac{\partial \mathcal{L}}{\partial W^{[2]}} &= (a^{[1]})^T \delta^{[2]} \\
\frac{\partial \mathcal{L}}{\partial b^{[2]}} &= \sum \delta^{[2]}
\end{align}
$$

**Hidden Layer Gradients:**

$$
\begin{align}
\delta^{[1]} &= \delta^{[2]} (W^{[2]})^T \odot f'^{[1]}(z^{[1]}) \\
\frac{\partial \mathcal{L}}{\partial W^{[1]}} &= x^T \delta^{[1]} \\
\frac{\partial \mathcal{L}}{\partial b^{[1]}} &= \sum \delta^{[1]}
\end{align}
$$

where $\odot$ denotes element-wise multiplication.

!!! important "Computational Efficiency"
    Backpropagation is efficient because it reuses computed gradients:

    - **Forward pass**: $O(W)$ where $W$ is the number of weights
    - **Backward pass**: $O(W)$ as well!

    Without backpropagation, computing gradients would require $O(W^2)$ operations.

### Implementation

```python
def backward(self, cache, y, learning_rate=0.01):
    """
    Backpropagation and parameter update.

    Parameters:
    -----------
    cache : dict
        Cached values from forward pass
    y : numpy.ndarray
        True labels of shape (n_samples, output_size)
    learning_rate : float
        Learning rate for gradient descent
    """
    m = y.shape[0]  # Number of samples

    # Retrieve cached values
    X = cache['X']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']

    # Output layer gradients
    dZ2 = A2 - y  # For binary cross-entropy with sigmoid
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients
    dA1 = dZ2 @ self.W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    # Update parameters
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2

# Add this method to the NeuralNetwork class
NeuralNetwork.backward = backward
```

## Loss Functions for Neural Networks

### Binary Cross-Entropy

For binary classification:

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
$$

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary cross-entropy loss.

    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted probabilities
    epsilon : float
        Small constant to avoid log(0)

    Returns:
    --------
    loss : float
        Average loss
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute loss
    loss = -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss
```

### Categorical Cross-Entropy

For multi-class classification with $C$ classes:

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

Used with **softmax** activation:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

```python
def softmax(z):
    """
    Softmax activation function.

    Parameters:
    -----------
    z : numpy.ndarray
        Pre-activation values

    Returns:
    --------
    probs : numpy.ndarray
        Probability distribution over classes
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

!!! tip "Numerical Stability"
    Always subtract the maximum value before computing exponentials in softmax to avoid overflow:

    $$
    \text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{C} e^{z_j - \max(z)}}
    $$

## Training a Neural Network

### Complete Training Loop

```python
def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
    """
    Train the neural network.

    Parameters:
    -----------
    X : numpy.ndarray
        Training data of shape (n_samples, input_size)
    y : numpy.ndarray
        Training labels of shape (n_samples, output_size)
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    verbose : bool
        Whether to print progress

    Returns:
    --------
    loss_history : list
        Loss values over epochs
    """
    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        y_pred, cache = self.forward(X)

        # Compute loss
        loss = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        # Backward pass
        self.backward(cache, y, learning_rate)

        # Print progress
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    return loss_history

# Add this method to the NeuralNetwork class
NeuralNetwork.train = train
```

### Example: XOR Problem

The XOR problem is a classic example that cannot be solved by a linear classifier:

```python
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])  # XOR labels

# Create and train network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
loss_history = nn.train(X, y, epochs=5000, learning_rate=0.5, verbose=True)

# Make predictions
predictions, _ = nn.forward(X)
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} → Predicted: {predictions[i][0]:.4f} (True: {y[i][0]})")

# Plot loss
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot decision boundary
plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
Z, _ = nn.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu',
           edgecolors='black', s=100)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Decision Boundary')
plt.colorbar(label='Predicted Probability')

plt.tight_layout()
plt.savefig('xor_neural_network.png')
plt.show()
```

## Advanced Topics

### Weight Initialization

Proper initialization is crucial for training deep networks.

#### Xavier/Glorot Initialization

For sigmoid/tanh activations:

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)
$$

```python
def xavier_initialization(n_in, n_out):
    """Xavier/Glorot initialization."""
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
```

#### He Initialization

For ReLU activations:

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)
$$

```python
def he_initialization(n_in, n_out):
    """He initialization for ReLU networks."""
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

!!! warning "Avoid Zero Initialization"
    Never initialize all weights to zero! This causes all neurons in a layer to learn the same function (symmetry problem).

### Batch Normalization

Normalizes activations across mini-batches:

$$
\hat{z} = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}
$$

Benefits:

- Accelerates training
- Allows higher learning rates
- Reduces sensitivity to initialization
- Has a regularizing effect

### Dropout

Randomly drops neurons during training with probability $p$:

```python
def dropout(A, keep_prob=0.5, training=True):
    """
    Apply dropout regularization.

    Parameters:
    -----------
    A : numpy.ndarray
        Activations
    keep_prob : float
        Probability of keeping a neuron
    training : bool
        Whether in training mode

    Returns:
    --------
    A : numpy.ndarray
        Activations with dropout applied
    """
    if training:
        mask = np.random.rand(*A.shape) < keep_prob
        A = A * mask / keep_prob  # Scale to maintain expected value
    return A
```

!!! note "Why Dropout Works"
    Dropout prevents co-adaptation of neurons, forcing the network to learn more robust features. It can be viewed as training an ensemble of networks.

## Universal Approximation Theorem

!!! important "Theoretical Foundation"
    **Universal Approximation Theorem**: A feedforward network with:

    - One hidden layer
    - Sufficient number of neurons
    - Non-linear activation function

    can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy.

This theorem provides the theoretical justification for using neural networks, though it doesn't tell us:

- How many neurons are needed
- How to find the optimal weights
- Whether the network will generalize well

## Summary

In this lecture, we covered:

- ✓ Perceptrons and activation functions (sigmoid, tanh, ReLU)
- ✓ Multi-layer perceptrons (MLPs) and forward propagation
- ✓ Backpropagation algorithm and gradient computation
- ✓ Loss functions (cross-entropy)
- ✓ Training neural networks end-to-end
- ✓ Advanced techniques (initialization, batch norm, dropout)
- ✓ Universal approximation theorem

!!! success "Congratulations!"
    You now understand the fundamentals of neural networks! These concepts form the foundation for deep learning architectures like CNNs, RNNs, and Transformers.

## Further Reading

- **Book**: "Deep Learning" by Goodfellow, Bengio, and Courville
- **Paper**: "Understanding the difficulty of training deep feedforward neural networks" by Glorot & Bengio
- **Online**: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
