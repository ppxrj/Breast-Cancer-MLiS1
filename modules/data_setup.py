import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
#from collections import Counter
!pip install ucimlrepo

#Dataset
from ucimlrepo import fetch_ucirepo
cancer_data = fetch_ucirepo(id=17) # fetch dataset
# data (as pandas dataframes)
X = cancer_data.data.features
y = cancer_data.data.targets
ids= cancer_data.data.ids

# Check the shape of data
print(f"Features shape: {X.shape}") #Features shape:  (569, 30)
print(f"Target shape: {y.shape}") #Target shape: (569, 1)
print(f"IDs shape: {ids.shape}") #IDs shape: (569, 1)
# Look at data
print(X.head())

print(y.head())
print(ids.head())
print (y.value_counts()) # Check target distribution: 357 benign (2), 212 malignant (4)
print(X.isnull().sum())# Check for missing values
print("ID CHECK")
print(ids.isnull().sum())# Check for missing values in ids column

# Basic statistics
print("DATA STATS")
print(X.info()) #all float
print(X.describe()) # count, mean, std, min, 25%, 50%, 75%, max for each feature

print(y.describe()) #569

print((X.isnull().sum()/ len(X))) #no empty values
duplicate_check=X.duplicated().any() # Check for duplicate rows
print(f"Are there duplicate rows? {duplicate_check}")