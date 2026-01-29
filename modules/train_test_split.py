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