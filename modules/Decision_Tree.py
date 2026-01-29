class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None): #all set to none
        self.feature = feature          # Feature index for splitting
        self.threshold = threshold      # Threshold value for splitting
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Class label for leaf nodes

class DecisionTree:

    """
    Decision tree classifier using information gain for splits.

    We're implementing CART (Breiman et al., 1984) with entropy-based splitting
    following Quinlan's approach (1986). The tree grows recursively, choosing the
    best split at each node by maximizing information gain.

    Parameters:
        max_depth: How deep the tree can grow (prevents overfitting)
        min_samples_split: Min samples needed to split a node
    """

    def __init__(self, max_depth=10, min_samples_split=2): #default values set to 10 questions
        """Initialize the tree with stopping criteria to prevent overfitting."""
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Build the tree by recursively finding the best splits."""
        self.root = self._grow_tree(X, y) #tree starts growing from root

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the tree. Stops when we hit max depth, run out of samples,
        or all samples belong to the same class.
        """
        A_samples, A_features = X.shape
        unique_classes = np.unique(y)

        # Stopping criteria
        if (len(unique_classes) == 1 or
            A_samples < self.min_samples_split or
            depth >= self.max_depth): #max set at 10
            leaf_value = self._mode_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, A_features)

        # If no valid split is found, create a leaf node
        if best_feature is None:
            leaf_value = self._mode_label(y)
            return Node(value=leaf_value)

        # Split the dataset
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y, A_features): #best feature and threshold to split on
        """
        Find the best feature and threshold to split on.

        We just try all features and all possible thresholds (greedy approach
        from Quinlan 1986) and pick whichever gives the highest information gain.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_index in range(A_features):
            X_column=X[:, feature_index] #store feature column
            thresholds = np.unique(X_column) #use stored column to get unique thresholds
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_threshold = threshold

        return split_idx, split_threshold

    #Information gain entropy
    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain from a split.

        Formula: IG = H(parent) - weighted_average(H(children))

        This measures how much splitting reduces uncertainty. Higher gain = better split.
        Based on Shannon entropy (1948), applied to decision trees by Quinlan (1986).
        """
        parent_entropy = self._entropy(y)
        # Generate split
        left_indices = X_column < threshold
        right_indices = X_column >= threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        #weighted avg of child entropies
        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        #info gain is parent entropy - child entropy
        info_gain= parent_entropy - child_entropy
        return info_gain

    def _entropy(self, y):
        """
        Calculate entropy: H(S) = -Σ(p_i * log2(p_i))

        Measures impurity/uncertainty in the data. Pure node (all same class) = 0 entropy.
        From Shannon (1948).
        """
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9)) #add small value to avoid log(0)
        return entropy

    def _mode_label(self, y):
        """Return the most frequent class label (for leaf nodes)."""
        values, counts = np.unique(y, return_counts=True)
        max_count_value= np.argmax(counts) #index of max count
        most_common = values[max_count_value]
        return most_common

    def predict(self, X):
        """Predict class labels by traversing the tree for each sample."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Recursively follow the decision rules until we hit a leaf."""
        if node.value is not None: #yes=leaf, no= decision node
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)