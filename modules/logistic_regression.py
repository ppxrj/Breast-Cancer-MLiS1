class LogisticRegression:
    """
    Logistic regression with gradient descent.

    Model: P(y=1|x) = σ(w^T x + b) where σ is the sigmoid function
    Loss: Cross-entropy with L2 regularization

    We use batch gradient descent to optimize. Following Chapter 4 of Hastie et al.
    (Elements of Statistical Learning, 2009) for the theory, with gradient formulas
    from Bishop (2006).

    Parameters:
        learning_rate: Step size α for gradient updates
        num_iterations: How many passes through the data
        regularization: L2 penalty λ to prevent overfitting
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=0.01):

        """Initialize logistic regression parameters"""

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.losses = []  # Track convergence

    def _sigmoid(self, z):
        """
        Sigmoid function: σ(z) = 1 / (1 + e^(-z))
        """
        #Squashes any input to a probability between 0 and 1

        # Clip to prevent overflow in exp()
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train using gradient descent.

        At each iteration:
            1. Compute predictions: ŷ = σ(Xw + b)
            2. Calculate gradients: ∂L/∂w = (1/n)X^T(ŷ - y) + λw
            3. Update: w ← w - α(∂L/∂w)

        Gradients derived from cross-entropy loss
        """

        n_samples, n_features = X.shape

        # Initialize parameters to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent optimization
        for iteration in range(self.num_iterations):
            # Forward pass: compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients (using calculus chain rule)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Add L2 regularization gradient
            dw += (self.regularization / n_samples) * self.weights

            # Update parameters (gradient descent step)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss for convergence monitoring
            if iteration % 100 == 0:
                loss = self._compute_loss(X, y)
                self.losses.append(loss)

    def _compute_loss(self, X, y):
        """
        Compute cross-entropy loss with L2 regularization.

        Loss Function Components:
            1. Cross-Entropy: Measures prediction error
               -1/n Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

            2. L2 Regularization: Prevents overfitting
               λ/(2n) ||w||²

        Derivation:
            From maximum likelihood estimation (MLE) of Bernoulli distribution
            Negative log-likelihood = cross-entropy

        Args:
            X: Features
            y: True labels

        Returns:
            Total loss value
        """
        n_samples = X.shape[0]
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)

        # Cross-entropy loss (avoid log(0) with epsilon)
        epsilon = 1e-9
        cross_entropy = -np.mean(
            y * np.log(y_predicted + epsilon) +
            (1 - y) * np.log(1 - y_predicted + epsilon)
        )

        # L2 regularization penalty
        l2_penalty = (self.regularization / (2 * n_samples)) * np.sum(self.weights ** 2)

        return cross_entropy + l2_penalty

    def predict_proba(self, X):
        """Return probability of positive class using sigmoid(w·x + b)."""

        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predict class labels: 1 if P(y=1) ≥ threshold, else 0."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

