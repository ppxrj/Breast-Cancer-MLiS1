# ENCODING
y_encoded= y.replace({2:0, 4:1}) # Encode target variable: 2 -> 0 (benign), 4 -> 1 (malignant)
#y_singleCol= y_encoded.values.ravel()) # Check encoded target values
print("Unique values",y_encoded.nunique()) # Check unique values in encoded target: [0, 1]
print("Value count",y_encoded.value_counts()) # Check encoded target distribution: 0- 357, 1- 212
print("Percentages",y_encoded.value_counts(normalize=True)*100) # Percentages: 0- 62.741652%, 1- 37.258348%
