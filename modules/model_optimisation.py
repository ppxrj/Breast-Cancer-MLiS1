# 1. Decision Tree Grid Search
print("\n1. DECISION TREE")
best_dt_params, dt_results = grid_search_dt(X_train_np, y_train_np, X_test_np, y_test_np)

# 2. KNN Grid Search
print("\n2. K-NEAREST NEIGHBORS")
best_knn_params, knn_results = grid_search_knn(X_train_np, y_train_np, X_test_np, y_test_np)

# 3. Logistic Regression Grid Search
print("\n 3. LOGISTIC REGRESSION")
best_lr_params, lr_results = grid_search_lr(X_train_np, y_train_np, X_test_np, y_test_np)

# 4. SVM Grid Search

print("\n4. SUPPORT VECTOR MACHINE")
best_svm_params, svm_results = grid_search_svm(X_train_np, y_train_np, X_test_np, y_test_np)

print("\nHYPERPARAMETER TUNING COMPLETE!")

# Summary of best parameters
print("\nBEST PARAMETERS FOUND:")
print(f"Decision Tree:        {best_dt_params}")
print(f"KNN:                 {best_knn_params}")
print(f"Logistic Regression:  {best_lr_params}")
print(f"SVM:                  {best_svm_params}")