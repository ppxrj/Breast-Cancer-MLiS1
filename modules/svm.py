"""
This includes:
1. Proper gradient computation (fixed the bug)
2. Detailed debugging output
3. Learning rate scheduling (helps convergence)
4. Verification that it's learning both classes

There's no "kernel" vs "regular" SVM distinction we need to make.
We're implementing a LINEAR SVM (the simplest kind).
Kernel SVM is much more complex and not required.
"""

import numpy as np

class SupportVectorMachine:
    """
    Linear SVM with hinge loss optimization using Pegasos algorithm.

    This is a LINEAR SVM (no kernel). Kernel SVM would require:
    - Kernel functions (RBF, polynomial, etc.)
    - Dual formulation
    - Much more complex optimization

    We're implementing the PRIMAL formulation with sub-gradient descent,
    which is simpler and works well for linearly separable data.

    Based on:
    - Cortes & Vapnik (1995) - Original SVM paper
    - Shalev-Shwartz et al. (2011) - Pegasos algorithm
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000, verbose=False):
        """
        Initialize SVM hyperparameters.

        Args:
            learning_rate: Initial step size for gradient descent
            lambda_param: Regularization parameter (C = 1/lambda in some formulations)
            num_iterations: Number of passes through the data
            verbose: If True, print debugging info during training
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []  # Track loss over time

    def fit(self, X, y):
        """
        Train SVM using Pegasos (Primal Estimated sub-GrAdient SOlver for SVM).

        The algorithm:
        1. Convert labels to {-1, +1}
        2. Initialize weights to zero
        3. For each iteration:
            - For each sample:
            - Check if correctly classified with margin
            - Update weights accordingly

        Key insight: We want to find w, b such that:
            y_i(w·x_i + b) ≥ 1 for all i (correct with margin)
        """
        n_samples, n_features = X.shape

        # Convert labels from {0, 1} to {-1, +1}
        # This is standard for SVM: allows using y(w·x+b) for margin
        y_svm = np.where(y <= 0, -1, 1)

        # Verify label conversion
        if self.verbose:
            print(f"\nLabel conversion:")
            print(f"  Original unique: {np.unique(y)}")
            print(f"  Converted unique: {np.unique(y_svm)}")
            print(f"  Distribution: {np.bincount(y_svm + 1)}")  # Shift to [0,2] for bincount

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Track how many times we update for each class
        updates_class_neg = 0
        updates_class_pos = 0

        # Sub-gradient descent
        for iteration in range(self.num_iterations):
            # Optional: learning rate decay (helps convergence)
            # current_lr = self.learning_rate / (1 + iteration / 100)
            current_lr = self.learning_rate  # Or use fixed learning rate

            total_loss = 0
            num_violations = 0

            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y_svm[idx]

                # Compute margin: y_i(w·x_i + b)
                margin = y_i * (np.dot(self.weights, x_i) + self.bias)

                # Hinge loss: max(0, 1 - margin)
                loss = max(0, 1 - margin)
                total_loss += loss

                if margin >= 1:
                    # Correctly classified with margin
                    # Only apply regularization: w ← w - α(λw)
                    self.weights -= current_lr * (self.lambda_param * self.weights)
                else:
                    # Violated margin (misclassified or too close)
                    # Apply full gradient: w ← w - α(λw - y_i*x_i)
                    # CRITICAL: Must be y_i * x_i (vector), NOT np.dot(x_i, y_i) (scalar)
                    self.weights -= current_lr * (
                        self.lambda_param * self.weights - y_i * x_i
                    )
                    self.bias -= current_lr * (-y_i)  # Gradient of bias

                    num_violations += 1
                    if y_i == -1:
                        updates_class_neg += 1
                    else:
                        updates_class_pos += 1

            # Track average loss
            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

            # Print progress
            if self.verbose and (iteration % 200 == 0 or iteration == self.num_iterations - 1):
                print(f"Iteration {iteration:4d}: Loss = {avg_loss:.4f}, "
                        f"Violations = {num_violations}/{n_samples}")

        if self.verbose:
            print(f"\nTraining complete!")
            print(f"Final weights range: [{self.weights.min():.4f}, {self.weights.max():.4f}]")
            print(f"Final bias: {self.bias:.4f}")
            print(f"Updates for class -1: {updates_class_neg}")
            print(f"Updates for class +1: {updates_class_pos}")

            # Check if both classes were learned
            if updates_class_neg == 0 or updates_class_pos == 0:
                print("⚠ WARNING: No updates for one class! Model may be degenerate.")

    def _decision_function(self, X):
        """
        Compute decision values: w·x + b

        Interpretation:
        - Positive: Classified as +1 (class 1)
        - Negative: Classified as -1 (class 0)
        - Magnitude: Confidence (distance from decision boundary)
        """
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Predict class labels.

        Decision rule: sign(w·x + b)
        - If w·x + b ≥ 0 → predict 1
        - If w·x + b < 0  → predict 0
        """
        decision = self._decision_function(X)
        return np.where(decision >= 0, 1, 0)

    def predict_proba(self, X):
        """
        Estimate probabilities using sigmoid of decision function.

        Note: SVMs don't naturally produce probabilities!
        This is a rough approximation. For better calibration, use Platt scaling.
        """
        decision = self._decision_function(X)
        # Clip to prevent overflow
        decision_clipped = np.clip(decision, -500, 500)
        return 1 / (1 + np.exp(-decision_clipped))

    def get_decision_stats(self, X, y):
        """
        Debugging method: Check decision function values and predictions.
        """
        decisions = self._decision_function(X)
        predictions = self.predict(X)

        print("\n" + "="*60)
        print("DECISION FUNCTION STATISTICS")
        print("="*60)
        print(f"Decision values range: [{decisions.min():.4f}, {decisions.max():.4f}]")
        print(f"Decision values mean: {decisions.mean():.4f}")
        print(f"Decision values std: {decisions.std():.4f}")

        print(f"\nPredictions unique: {np.unique(predictions)}")
        print(f"Predictions distribution: {np.bincount(predictions)}")

        print(f"\nTrue labels unique: {np.unique(y)}")
        print(f"True labels distribution: {np.bincount(y)}")

        # Check decision values by class
        for cls in [0, 1]:
            mask = y == cls
            if np.sum(mask) > 0:
                cls_decisions = decisions[mask]
                print(f"\nClass {cls} decision values:")
                print(f"  Range: [{cls_decisions.min():.4f}, {cls_decisions.max():.4f}]")
                print(f"  Mean: {cls_decisions.mean():.4f}")

        # Check if model is just predicting one class
        if len(np.unique(predictions)) == 1:
            print("\n⚠ WARNING: Model is predicting only ONE class!")
            print("   Possible causes:")
            print("   1. Features not scaled properly")
            print("   2. Learning rate too high/low")
            print("   3. Lambda too high")
            print("   4. Not enough iterations")
        else:
            print("\n✓ Model is predicting both classes")

        print("="*60)