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
