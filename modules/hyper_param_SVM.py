def grid_search_svm(X_train, y_train, X_test, y_test):
    """
    Grid search for SVM hyperparameters.
    """
    best_acc = 0
    best_param = {}
    results = []

    learning_rates = [0.0001, 0.001, 0.01]
    lambda_params = [0.001, 0.01, 0.1, 1.0]
    num_iterations = [500, 1000, 2000]

    total_combo = len(learning_rates) * len(lambda_params) * len(num_iterations)
    print(f"\nTesting {total_combo} parameter combinations for SVM...")

    combo_count = 0
    for lr in learning_rates:
        for lam in lambda_params:
            for iters in num_iterations:
                combo_count += 1
                print(f"[{combo_count}/{total_combo}] lr={lr}, lambda={lam}, iters={iters}", end='')

                model = SupportVectorMachine(learning_rate=lr, lambda_param=lam, num_iterations=iters)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, verbose=False)

                results.append({
                    'learning_rate': lr,
                    'lambda_param': lam,
                    'num_iterations': iters,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                })

                if metrics['accuracy'] > best_acc:
                    best_acc = metrics['accuracy']
                    best_param = {'learning_rate': lr, 'lambda_param': lam, 'num_iterations': iters}
                    print(f" -> New best! Accuracy: {best_acc*100:.2f}%")
                else:
                    print(f" -> Accuracy: {metrics['accuracy']*100:.2f}%")

    print(f"\nBest parameters found: {best_param}")
    print(f"Best accuracy: {best_acc*100:.2f}%")

    return best_param, results