import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

data = pd.read_csv("/adult.data", names=column_names, na_values=" ?")
#print(data.shape) (32561, 15)

print("Before Cleaning\nNumber of missing values per column:\n", data.isnull().sum())

data["workclass"] = data["workclass"].fillna("Private")  #found that Private was the mode with 22696 values so we are replacing the missing values with Private
data["occupation"] = data["occupation"].fillna("Prof-specialty") #found that Occupation mode was Prof-specialty
data["native-country"] = data["native-country"].fillna("United-States") #Mode is United States, so replace it with United States
print("\nAfter Cleaning\nNumber of missing values per column:\n", data.isnull().sum())  #No more missing values
#print(data.dtypes)
data['income'] = data['income'].str.strip()

data_encoded = pd.get_dummies(data, columns=[
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
], drop_first=True)
data_encoded['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

#print(data_encoded.head())
#print(data_encoded.shape)
#income_counts = data_encoded["income"].value_counts()
#print(income_counts)
#output_file = '/Users/leobaltazar/Desktop/Data Mining/modified_adult.csv'
#data_encoded.to_csv(output_file, index=False) Used this to see how the data looked
X = data_encoded.drop(columns=['income'])
y = data_encoded['income']
# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}") # verify correct splitting
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Apply SMOTE to balance dataset
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("\nBefore SMOTE:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_train_resampled).value_counts()) # both classes have the same amount of data.

param_grid = {
    'n_estimators': [50, 100, 200],         # Num of trees in the forest
    'max_depth': [10, 20, None],            # Max depth of the tree
    'min_samples_split': [2, 5, 10],        # Min samples required to split a node
    'min_samples_leaf': [1, 2, 4]           # Min samples required at a leaf node
}

#Initialize Random Forest model
rf = RandomForestClassifier(random_state=42)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_) # will display best Hyperparameters
best_params = grid_search.best_params_

# Train Random Forest model with optimized hyperparameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train_resampled, y_train_resampled)

print("Final Random Forest model trained with optimized hyperparameters.")

# Make predictions on test set
y_pred = best_rf.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}") # was always around ~85%

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix) # will display the conf matrix

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['<=50K', '>50K'])
disp.plot(cmap='Reds', values_format='d')
plt.title("Confusion Matrix")
plt.show()

feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns) #will find the features that have importance
top_features = feature_importances.sort_values(ascending=False).head(10) #sorts so we can display


# Plot top 10 important features
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index, palette="viridis", legend=False)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features Influencing Income Classification")
plt.show()

"""
Best Hyperparameters: {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
Final Random Forest model trained with optimized hyperparameters.
Test Set Accuracy: 0.8686
"""