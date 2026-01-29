print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Model Accuracy Comparison
ax1 = axes[0, 0]
models = comparison_df['Model']
accuracies = comparison_df['Accuracy']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
ax1.set_ylim([0.85, 1.0])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. All Metrics Grouped Bar Chart
ax2 = axes[0, 1]
x = np.arange(len(models))
width = 0.2
ax2.bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy', color='#2E86AB', edgecolor='black')
ax2.bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision', color='#A23B72', edgecolor='black')
ax2.bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall', color='#F18F01', edgecolor='black')
ax2.bar(x + 1.5*width, comparison_df['F1-Score'], width, label='F1-Score', color='#C73E1D', edgecolor='black')
ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
ax2.set_title('All Metrics Comparison', fontweight='bold', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.legend(loc='lower right')
ax2.set_ylim([0.85, 1.0])
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 3. Confusion Matrix - Best Model
ax3 = axes[0, 2]
if best_model_name == 'Decision Tree':
    best_metrics = metrics_dt
elif best_model_name == 'KNN':
    best_metrics = metrics_knn
elif best_model_name == 'Logistic Regression':
    best_metrics = metrics_lr
else:
    best_metrics = metrics_svm

cm = np.array([[best_metrics['tn'], best_metrics['fp']],
               [best_metrics['fn'], best_metrics['tp']]])
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax3,
           cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 16, 'fontweight': 'bold'},
           linewidths=2, linecolor='black')
ax3.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
ax3.set_ylabel('True Label', fontweight='bold', fontsize=12)
ax3.set_title(f'Confusion Matrix: {best_model_name}', fontweight='bold', fontsize=14)
ax3.set_xticklabels(['Benign (0)', 'Malignant (1)'])
ax3.set_yticklabels(['Benign (0)', 'Malignant (1)'])

# 4. F1-Score Ranking
ax4 = axes[1, 0]
sorted_df = comparison_df.sort_values('F1-Score')
colors_rank = ['#E63946' if x < 0.94 else '#F77F00' if x < 0.96 else '#06A77D'
              for x in sorted_df['F1-Score']]
bars = ax4.barh(sorted_df['Model'], sorted_df['F1-Score'],
               color=colors_rank, edgecolor='black', linewidth=2, alpha=0.8)
ax4.set_xlabel('F1-Score', fontweight='bold', fontsize=12)
ax4.set_title('Model Ranking by F1-Score', fontweight='bold', fontsize=14)
ax4.set_xlim([0.85, 1.0])
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax4.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

# 5. Precision vs Recall
ax5 = axes[1, 1]
ax5.scatter(comparison_df['Recall'], comparison_df['Precision'],
           s=300, c=colors, edgecolors='black', linewidth=2, alpha=0.7)
for i, model in enumerate(comparison_df['Model']):
    ax5.annotate(model,
                (comparison_df['Recall'].iloc[i], comparison_df['Precision'].iloc[i]),
                xytext=(5, 5), textcoords='offset points',
                fontweight='bold', fontsize=9)
ax5.set_xlabel('Recall', fontweight='bold', fontsize=12)
ax5.set_ylabel('Precision', fontweight='bold', fontsize=12)
ax5.set_title('Precision vs Recall Trade-off', fontweight='bold', fontsize=14)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.set_xlim([0.85, 1.0])
ax5.set_ylim([0.85, 1.0])

# 6. Error Breakdown
ax6 = axes[1, 2]
error_types = ['TN', 'FP', 'FN', 'TP']
error_values = [best_metrics['tn'], best_metrics['fp'],
               best_metrics['fn'], best_metrics['tp']]
colors_error = ['#06A77D', '#F77F00', '#E63946', '#2E86AB']
bars = ax6.bar(error_types, error_values, color=colors_error,
              edgecolor='black', linewidth=2, alpha=0.8)
ax6.set_ylabel('Count', fontweight='bold', fontsize=12)
ax6.set_title(f'Confusion Matrix Breakdown: {best_model_name}',
             fontweight='bold', fontsize=14)
ax6.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels and descriptions
labels_desc = ['True\nNegative', 'False\nPositive', 'False\nNegative', 'True\nPositive']
for i, (bar, desc) in enumerate(zip(bars, labels_desc)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{int(height)}', ha='center', va='bottom',
            fontweight='bold', fontsize=12)
    ax6.text(bar.get_x() + bar.get_width()/2., -10,
            desc, ha='center', va='top', fontsize=8)

plt.suptitle('Breast Cancer Classification - Comprehensive Model Analysis',
            fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()