print("MODEL COMPARISON and PERFORMANCE SUMMARY")

# Create comparison dataframe
comparison_data = {
    'Model': ['Decision Tree', 'KNN', 'Logistic Regression', 'SVM'],
    'Accuracy': [
        metrics_dt['accuracy'],
        metrics_knn['accuracy'],
        metrics_lr['accuracy'],
        metrics_svm['accuracy']
    ],
    'Precision': [
        metrics_dt['precision'],
        metrics_knn['precision'],
        metrics_lr['precision'],
        metrics_svm['precision']
    ],
    'Recall': [
        metrics_dt['recall'],
        metrics_knn['recall'],
        metrics_lr['recall'],
        metrics_svm['recall']
    ],
    'F1-Score': [
        metrics_dt['f1_score'],
        metrics_knn['f1_score'],
        metrics_lr['f1_score'],
        metrics_svm['f1_score']
    ],
    'Best Parameters': [
        str(best_dt_params),
        str(best_knn_params),
        str(best_lr_params),
        str(best_svm_params)
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n")
print(comparison_df.to_string(index=False))

# Identify best model
best_idx = comparison_df['F1-Score'].idxmax()
best_model_name = comparison_df.loc[best_idx, 'Model']
best_accuracy = comparison_df.loc[best_idx, 'Accuracy']
best_f1 = comparison_df.loc[best_idx, 'F1-Score']


print("\nBEST PERFORMING MODEL")

print(f"Model:     {best_model_name}")
print(f"Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"F1-Score:  {best_f1:.4f} ({best_f1*100:.2f}%)")