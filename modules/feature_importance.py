import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def extract_feature_importance_recursive(tree, feature_names):
    """Extract feature importance from decision tree by counting usage."""
    importance_dict = {feature: 0 for feature in feature_names}

    def traverse(node, depth=0):
        if node.value is not None:  # Leaf node
            return

        feature_name = feature_names[node.feature]
        importance_dict[feature_name] += 1

        if node.left:
            traverse(node.left, depth + 1)
        if node.right:
            traverse(node.right, depth + 1)

    traverse(tree.root)

    # Normalize
    total = sum(importance_dict.values())
    if total > 0:
        for key in importance_dict:
            importance_dict[key] /= total

    return importance_dict

# Get feature names
feature_names = X_train.columns.tolist()

# Calculate importance
importance = extract_feature_importance_recursive(dt, feature_names)
importance_df = pd.DataFrame(list(importance.items()),
                             columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values('Importance', ascending=False)
importance_df = importance_df[importance_df['Importance'] > 0]  # Only used features

print("\n")
print(importance_df.to_string(index=False))

# Visualize
plt.figure(figsize=(12, 6))
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'],
        color=colors_feat, edgecolor='black', linewidth=1.5)
plt.xlabel('Relative Importance', fontweight='bold', fontsize=12)
plt.title('Feature Importance - Decision Tree Analysis',
         fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{width:.3f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n Feature importance analysis complete!")
print("\n Top 3 Most Important Features:")
for i, row in importance_df.head(3).iterrows():
    print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
