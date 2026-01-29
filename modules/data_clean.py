print((X.isnull().sum()/ len(X))) #2.289% missing values in 'Bare_nuclei' column
#print(X['Bare_nuclei'].value_counts()) # Check unique values in 'Bare_nuclei' column: 1.0 402, 10.0 132

# Missing values
X_null=X.fillna(X.mean()) # Fill missing values with mean of the column, mean since it's numerical
#X_null['Bare_nuclei'] = X_null['Bare_nuclei'].astype(int) # Convert 'Bare_nuclei' to integer type

#Duplicate rows
duplicate_check=X_null.duplicated().any() # Check for duplicate rows
print(f"Are there duplicate rows? {duplicate_check}")
print(f"Shape before cleaning: {X_null.shape}") # Shape before cleaning (699, 9)
print(f"Number of duplicate rows: {X_null.duplicated().sum()}") # Number of duplicate rows, 237
dup=X_null[X_null.duplicated(keep=False)]  # Display duplicate rows

X_clean=X_null.drop_duplicates() # Drop duplicate rows if any
print(f"New shape after cleaning: {X_clean.shape}") # New shape after cleaning (462, 9)
y_clean=y.loc[X_clean.index] # Align target variable with cleaned features
ids_clean=ids.loc[X_clean.index] # Align ids with cleaned features
print(f"Target shape after cleaning: {y_clean.shape}") # Target shape after cleaning
print(f"IDs shape after cleaning: {ids_clean.shape}") # IDs shape after cleaning

print("DATA CLEANING CHECK")
print(X_clean.shape)
print(X_clean.columns)
print(X_clean.dtypes)
print(X_clean.isnull().sum())
print(y_clean.value_counts())