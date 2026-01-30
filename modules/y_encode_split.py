# ENCODING
y_encoded= y.replace({2:0, 4:1}) # Encode target variable: 2 -> 0 (benign), 4 -> 1 (malignant)
#y_singleCol= y_encoded.values.ravel()) # Check encoded target values
print("Unique values",y_encoded.nunique()) # Check unique values in encoded target: [0, 1]
print("Value count",y_encoded.value_counts()) # Check encoded target distribution: 0- 357, 1- 212
print("Percentages",y_encoded.value_counts(normalize=True)*100) # Percentages: 0- 62.741652%, 1- 37.258348%

import pandas as pd
import numpy as np
from scipy import stats

#Plan:class separation, shuffle in each class, split into class, merge, shuffle merged
np.random.seed(42)

# Separate classes
benign_indices = y_encoded[y_encoded == 0].index.tolist()
malignant_indices = y_encoded[y_encoded == 1].index.tolist()

# Shuffle indices within each class
np.random.shuffle(benign_indices)
np.random.shuffle(malignant_indices)

# Split indices into train and test sets (80-20 split)
test_size=0.2 #80/20 but could try 70/30
benign_split = int(len(benign_indices)*(1 - test_size))
print("Number of indices benign", benign_split)
malignant_split = int(len(malignant_indices)*(1 - test_size))
print("Number of malignant", malignant_split)

benign_train_indices = benign_indices[:benign_split] #80% for training
benign_test_indices = benign_indices[benign_split:] #20% for testing

malignant_train_indices = malignant_indices[:malignant_split]
malignant_test_indices = malignant_indices[malignant_split:]


#Merge train and test indices
train_set= benign_train_indices + malignant_train_indices
test_set= benign_test_indices + malignant_test_indices

# Shuffle merged train and test sets
np.random.shuffle(train_set)
np.random.shuffle(test_set)

X_train= X.loc[train_set]
y_train= y_encoded.loc[train_set]
X_test= X.loc[test_set]
y_test= y_encoded.loc[test_set]
print("Train shape:", X_train.shape, y_train.shape) #(910, 30) (910, 1)
print("Test shape:", X_test.shape, y_test.shape) #(228, 30) (228, 1)

#CODE CHECK
print("X set", X_train.shape, X_test.shape)
print("Y set", y_train.shape, y_test.shape)
print("Training distribution",y_train.value_counts(), y_train.value_counts(normalize=True)*100)
print("Testing distribution",y_test.value_counts(), y_test.value_counts(normalize=True)*100)

# SCALING
train_mean= X_train.mean()
train_std= X_train.std()
print("Train mean", train_mean)
print("Train std", train_std)

X_train_scaled= (X_train - train_mean)/ train_std
X_test_scaled= (X_test - train_mean)/ train_std # standarised using train mean and std

# Convert scaled pandas DataFrames to NumPy arrays
X_train_np = X_train_scaled.values
X_test_np  = X_test_scaled.values

y_train_np = y_train.values.ravel()
y_test_np  = y_test.values.ravel()

print("Scaled training set", X_train_scaled.mean())
print("Scaled testing set", X_test_scaled.mean())
print("Scaled training set std", X_train_scaled.std())  #all std=1
print("Scaled testing set std", X_test_scaled.std())  #close to 1
print("Shape", X_train_scaled.shape, X_test_scaled.shape)