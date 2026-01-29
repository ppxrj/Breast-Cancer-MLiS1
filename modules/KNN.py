class KNearestNeighbors:
    """
    KNN classifier.

    Instead of training a model, it just stores the training data and makes predictions by finding the k closest examples.
    Classification is by majority vote.

    We support Euclidean and Manhattan distance metrics.

    Parameters:
        k: Number of neighbors to consider
        distance_metric: 'euclidean' or 'manhattan'
    """

    def __init__(self, k=5, distance_metric='euclidean'):
        """Set k and distance metric."""
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Just store the training data. KNN doesn't actually "train" anything.
        All the work happens during prediction.
        """
        self.X_train = X
        self.y_train = y

    def _compute_distance(self, x1, x2):
        """
        Calculate distance between two points.

        Euclidean: d(x,y) = √(Σᵢ(xᵢ - yᵢ)²) - straight line distance
        Manhattan: d(x,y) = Σᵢ|xᵢ - yᵢ| - grid/taxicab distance
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Distance metric must be 'euclidean' or 'manhattan'")

    def _get_neighbors(self, x):
        """
        Find KNN by computing all distances and taking the k smallest.

        This is a naive O(n) search.
        """
        # Calculate distances to all training samples
        distances = [self._compute_distance(x, x_train)
                    for x_train in self.X_train]

        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]

        return k_indices

    def _majority_vote(self, neighbor_labels):
        """
        Find the most common class among the k neighbors. Simple majority wins.

        We considered distance-weighted voting but it didn't improve results
        on our dataset, so stuck with the simpler approach.
        """
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        max_count_idx = np.argmax(counts)
        return unique_labels[max_count_idx]

    def predict(self, X):
        """Predict by finding k nearest neighbors for each sample and taking majority vote."""
        predictions = []

        for x in X:
            # Find k nearest neighbors
            k_indices = self._get_neighbors(x)

            # Get labels of neighbors
            k_nearest_labels = self.y_train[k_indices]

            # Majority vote
            prediction = self._majority_vote(k_nearest_labels)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Estimate P(y=1) as the proportion of positive neighbors: (# positive) / k
        Simple frequency-based probability.
        """
        probabilities = []

        for x in X:
            k_indices = self._get_neighbors(x)
            k_nearest_labels = self.y_train[k_indices]

            # Probability = proportion of positive class
            prob_positive = np.sum(k_nearest_labels == 1) / self.k
            probabilities.append(prob_positive)

        return np.array(probabilities)