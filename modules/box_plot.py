import matplotlib.pyplot as plt
import numpy as np
feature_list=['radius','texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
fig, axes= plt.subplots(5,2, figsize=(12,12))
axes= axes.flatten()

for i, feature in enumerate(feature_list):
  col_plot=[f'{feature}1', f'{feature}2', f'{feature}3']
  X[col_plot].boxplot(ax=axes[i])
  axes[i].set_title(f'{feature.capitalize()}- All 3 measurements', fontsize=12)
  axes[i].set_ylabel('Value')
  axes[i].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()