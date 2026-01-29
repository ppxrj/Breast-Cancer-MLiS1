# RESULTS TABLE WITH FIXED SVM

print("Training all models with final parameters...\n")

results = []

# 1. Decision Tree
print("1. Training Decision Tree...")
dt = DecisionTree(max_depth=10, min_samples_split=2)
dt.fit(X_train_np, y_train_np)
y_pred_dt = dt.predict(X_test_np)
metrics_dt = calculate_metrics(y_test_np, y_pred_dt, verbose=False)

# Just add the entire metrics dict (it already has the right keys!)
metrics_dt['Model'] = 'Decision Tree'
results.append(metrics_dt)

# 2. KNN
print("2. Training KNN...")
knn = KNearestNeighbors(k=5, distance_metric='euclidean')
knn.fit(X_train_np, y_train_np)
y_pred_knn = knn.predict(X_test_np)
metrics_knn = calculate_metrics(y_test_np, y_pred_knn, verbose=False)

metrics_knn['Model'] = 'KNN'
results.append(metrics_knn)

# 3. Logistic Regression
print("3. Training Logistic Regression...")
lr = LogisticRegression(best-prm)
lr.fit(X_train_np, y_train_np)
y_pred_lr = lr.predict(X_test_np)
metrics_lr = calculate_metrics(y_test_np, y_pred_lr, verbose=False)

metrics_lr['Model'] = 'Logistic Regression'
results.append(metrics_lr)

# 4. SVM - WITH FIXED CODE
print("4. Training SVM (with gradient fix)...")
svm = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, num_iterations=2000)
svm.fit(X_train_np, y_train_np)
y_pred_svm = svm.predict(X_test_np)
metrics_svm = calculate_metrics(y_test_np, y_pred_svm, verbose=False)

metrics_svm['Model'] = 'SVM'
results.append(metrics_svm)

# Create DataFrame
results_df = pd.DataFrame(results)

# Reorder columns to put Model first
columns_order = ['Model', 'accuracy', 'precision', 'recall', 'f1_score', 'tn', 'fp', 'fn', 'tp']
results_df = results_df[columns_order]

print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)
print(results_df.to_string(index=True))
print("="*80)

# Display the DataFrame
results_df