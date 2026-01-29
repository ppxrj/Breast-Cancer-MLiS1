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
