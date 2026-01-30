print("\nChecking and encoding labels...")
print(f"Current labels: {np.unique(y_train_np)}")

# Encode: M → 1 (malignant), B → 0 (benign)
y_train_np = np.where(y_train_np == 'M', 1, 0)
y_test_np = np.where(y_test_np == 'M', 1, 0)

print(f"Encoded labels: {np.unique(y_train_np)}")
print(f"Mapping: M → 1, B → 0")
print("Labels encoded successfully!\n")

def grid_search_dt(X_train, y_train, X_test, y_test):
    """
    Try different combinations of hyperparameters to find the best one for Decision Tree.
    It just brute force through all combinations.
    """
    best_acc = 0
    best_param = {}
    results = []

    max_depth = [2, 4, 6, 8, 10, 15, 20, 25, 30]  # different tree depths to try
    min_samples_split = [2, 5, 10, 15, 20]  # minimum samples needed to split

    total_combo = len(max_depth) * len(min_samples_split)
    print(f"\nTesting {total_combo} parameter combinations for Decision Tree...")

    combo_count = 0
    for depth in max_depth:
        for min_samples in min_samples_split:
            combo_count += 1
            print(f"[{combo_count}/{total_combo}] Testing: max_depth={depth}, min_samples_split={min_samples}", end='')

            dtree = DecisionTree(max_depth=depth, min_samples_split=min_samples)
            dtree.fit(X_train, y_train)
            y_pred = dtree.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, verbose=False )

            results.append({
                'max_depth': depth,
                'min_samples_split': min_samples,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })

            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_param = {'max_depth': depth, 'min_samples_split': min_samples}
                print(f" -> New best! Accuracy: {best_acc*100:.2f}%")
            else:
                print(f" -> Accuracy: {metrics['accuracy']*100:.2f}%")

    print(f"\nBest parameters found: {best_param}")
    print(f"Best accuracy: {best_acc*100:.2f}%")

    return best_param, results

def grid_search_knn(X_train, y_train, X_test, y_test):
    #Grid search for KNN to find optimal k and distance metric.
    best_acc = 0
    best_param = {}
    results = []

    k_values = [1, 3, 5, 7, 9, 11, 15, 19]
    distance_metrics = ['euclidean', 'manhattan']

    total_combo = len(k_values) * len(distance_metrics)
    print(f"\nTesting {total_combo} parameter combinations for KNN...")

    combo_count = 0
    for k in k_values:
        for metric in distance_metrics:
            combo_count += 1
            print(f"[{combo_count}/{total_combo}] Testing: k={k}, metric={metric}", end='')

            knn = KNearestNeighbors(k=k, distance_metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, verbose=False)

            results.append({
                'k': k,
                'distance_metric': metric,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })

            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_param = {'k': k, 'distance_metric': metric}
                print(f" -> New best! Accuracy: {best_acc*100:.2f}%")
            else:
                print(f" -> Accuracy: {metrics['accuracy']*100:.2f}%")

    print(f"\nBest parameters found: {best_param}")
    print(f"Best accuracy: {best_acc*100:.2f}%")

    return best_param, results
