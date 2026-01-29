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