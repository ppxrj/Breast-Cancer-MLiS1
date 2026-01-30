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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Regression= scatter plot
feature_list=['radius','texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
fig, axes= plt.subplots(10,3, figsize=(15,25))
#axes=axes.flatten()

for i, feature in enumerate(feature_list):
    for j in range(1,4):
        column=f'{feature}{j}'
        axes[i, j-1].scatter(range(len(X)), X[column], alpha=0.5)

        if i==9:
            axes[i, j-1].set_xlabel('Index')
        if j==1:
            axes[i, j-1].set_ylabel(f'{feature}')

        axes[i, j-1].set_title(f'Scatter Plot of {column} -Outlier Detection')

plt.tight_layout()
plt.show() #out of loop to see all graphs together

#Histogram for all features to find outliers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

fig, axes= plt.subplots(10,3, figsize=(18,30))
axes= axes. flatten()

for i, column in enumerate(X.columns):
    ax=axes[i]
    mean = X[column].mean()
    std = X[column].std()
    median= X[column].median()
    x= np.linspace(X[column].min(), X[column].max(), 100)

    sns.histplot(X[column], bins=30, kde=True, color='blue', edgecolor='black', ax=ax, stat= 'density', label='Histogram', line_kws={'color': 'purple', 'lw': 2, 'label': 'Actual Distribution (KDE)'})
    ax.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    ax.axvline(median, color='green', linestyle='dashed', linewidth=2, label='Median')
    ax.axvline(mean + 3*std, color='orange', linestyle='dashed', linewidth=2, label='Mean + 3*Std Dev')
    ax.axvline(mean - 3*std, color='orange', linestyle='dashed', linewidth=2, label='Mean - 3*Std Dev')

    normal_curve= stats.norm.pdf(x, mean, std)
    ax.plot(x, normal_curve, color='red', linewidth=2, label='Normal Distribution Fit')
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()