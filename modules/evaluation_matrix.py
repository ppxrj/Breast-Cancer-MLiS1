def calculate_metrics(y_true, y_pred, model_name=None, verbose=True):
    """
    Calculate standard classification metrics from scratch.

    Metrics:
        - Accuracy: (TP + TN) / Total
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1: 2 * (Precision * Recall) / (Precision + Recall)

    We compute these manually to show understanding of the evaluation process.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Optional name for printing
        verbose: If True, print the metrics (default: True)

    Returns:
        Dictionary with all metrics and confusion matrix values
    """
    # Calculate confusion matrix components
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Calculate metrics
    total = len(y_true)
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results only if verbose=True
    if verbose:
        print(f"\nEVALUATION RESULTS: {model_name if model_name else 'Model'}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:6.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:6.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:6.2f}%)")
        print(f"  F1-Score:  {f1_score:.4f} ({f1_score*100:6.2f}%)")

        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                0       1")
        print(f"  Actual  0    {true_negative:3d}    {false_positive:3d}   (TN)  (FP)")
        print(f"          1    {false_negative:3d}    {true_positive:3d}   (FN)  (TP)")

    return {
        "Model": model_name if model_name else "Model",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive
    }

print("Metrics function defined")

# Quick debug - run to check keys exist
print("Checking calculate_metrics return values...")
test_metrics = calculate_metrics(y_test_np[:10], y_test_np[:10], verbose=False)
print(f"Keys returned: {list(test_metrics.keys())}")