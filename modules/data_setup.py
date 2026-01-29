#Dataset
from ucimlrepo import fetch_ucirepo
cancer_data = fetch_ucirepo(id=17) # fetch dataset
# data (as pandas dataframes)
X = cancer_data.data.features
y = cancer_data.data.targets
ids= cancer_data.data.ids

# Check the shape of data
print(f"Features shape: {X.shape}") #Features shape: (699, 9)
print(f"Target shape: {y.shape}") #Target shape: (699, 1)
print(f"IDs shape: {ids.shape}") #IDs shape: (699, 1)
# Look at data
print(X.head())

print(y.head())
print(ids.head())
print (y.value_counts()) # Check target distribution: 458 benign (2), 241 malignant (4)
print(X.isnull().sum())# Check for missing values, 16 missing values in 'Bare_nuclei' column
print("ID CHECK")
print(ids.isnull().sum())# Check for missing values in ids column

# Basic statistics
print("DATA STATS")
print(X.info()) #all int, bare_nuclei float (683 entries non-null)
print(X.describe()) # count, mean, std, min, 25%, 50%, 75%, max for each feature
print(y.describe())