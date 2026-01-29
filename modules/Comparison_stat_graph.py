# Class Distribution
print(y.value_counts(normalize=True)*100) # Class distribution percentages: B- 62.742%, M 37.258%
plt.figure(figsize=(6,4))
y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution of Target Variable')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0,1], labels=['Benign (2)', 'Malignant (4)'], rotation=0)
plt.tight_layout()
plt.show()

# Correlation matrix to see relationships between features
plt.figure(figsize=(10,8))
correlation_matrix = X.corr() #correlation matrix
#print(correlation_matrix)
sns.heatmap(correlation_matrix, fmt=".2f", cmap='coolwarm', square=True, linecolor='white', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()