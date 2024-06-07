#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# Define features and target variable
X = df.drop('year', axis=1)  # Replace 'target_column' with the actual target column name
y = df['year']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=10000)),
    'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
    'Random Forest': OneVsRestClassifier(RandomForestClassifier(random_state=42)),
    'Gradient Boosting': OneVsRestClassifier(GradientBoostingClassifier(random_state=42)),
    'Bagging': OneVsRestClassifier(BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)),
    'K-Nearest Neighbors': OneVsRestClassifier(KNeighborsClassifier())
}

# Define a function to evaluate models and store results
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = "ROC AUC score not applicable"

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}')
    print(f'{model_name} - Mean Squared Error (MSE): {mse}, Mean Absolute Error (MAE): {mae}')
    
    cm = confusion_matrix(y_test, y_pred)
   
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'MSE': mse,
        'MAE': mae
    }

# Train and evaluate models
results = []
for model_name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    result = evaluate_model(model, X_test_preprocessed, y_test, model_name)
    results.append(result)

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print(results_df)

# Find the best model based on accuracy
best_model_info = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Model based on Accuracy:")
print(best_model_info)

# Save the best model and preprocessor
best_model_name = best_model_info['Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print(f"{best_model_name} and preprocessor saved as best_model.pkl and preprocessor.pkl")

# Load the best model and preprocessor
loaded_model = joblib.load('best_model.pkl')
loaded_preprocessor = joblib.load('preprocessor.pkl')
print(f"{best_model_name} and preprocessor loaded successfully")


# In[2]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')  # Update with the correct path

# Randomly pick 20 data points from the dataset
df_sample = df.sample(n=20, random_state=42)

# Define features and target variable
X_sample = df_sample.drop('year', axis=1)  # Replace 'target_column' with the actual target column name
y_sample = df_sample['year']

# Load the saved preprocessor and model
loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_model = joblib.load('best_model.pkl')
print("Model and preprocessor loaded successfully")

# Preprocess the sample data
X_sample_preprocessed = loaded_preprocessor.transform(X_sample)

# Apply the model to the subset and evaluate the performance
y_sample_pred = loaded_model.predict(X_sample_preprocessed)

# Calculate performance metrics
accuracy = accuracy_score(y_sample, y_sample_pred)
precision = precision_score(y_sample, y_sample_pred, average='weighted')
recall = recall_score(y_sample, y_sample_pred, average='weighted')
f1 = f1_score(y_sample, y_sample_pred, average='weighted')

try:
    roc_auc = roc_auc_score(y_sample, y_sample_pred, multi_class='ovr')
except ValueError:
    roc_auc = "ROC AUC score not applicable"

mse = mean_squared_error(y_sample, y_sample_pred)
mae = mean_absolute_error(y_sample, y_sample_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Confusion Matrix:\n{confusion_matrix(y_sample, y_sample_pred)}')

# Plot the confusion matrix
cm = confusion_matrix(y_sample, y_sample_pred)


# In[3]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# Define features and target variable
X = df.drop('selling_price', axis=1)  # Replace 'target_column' with the actual target column name
y = df['selling_price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=10000)),
    'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
    'Random Forest': OneVsRestClassifier(RandomForestClassifier(random_state=42)),
    'Gradient Boosting': OneVsRestClassifier(GradientBoostingClassifier(random_state=42)),
    'Bagging': OneVsRestClassifier(BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)),
    'K-Nearest Neighbors': OneVsRestClassifier(KNeighborsClassifier())
}

# Define a function to evaluate models and store results
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = "ROC AUC score not applicable"

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}')
    print(f'{model_name} - Mean Squared Error (MSE): {mse}, Mean Absolute Error (MAE): {mae}')
    
    cm = confusion_matrix(y_test, y_pred)
   
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'MSE': mse,
        'MAE': mae
    }

# Train and evaluate models
results = []
for model_name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    result = evaluate_model(model, X_test_preprocessed, y_test, model_name)
    results.append(result)

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print(results_df)

# Find the best model based on accuracy
best_model_info = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Model based on Accuracy:")
print(best_model_info)

# Save the best model and preprocessor
best_model_name = best_model_info['Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print(f"{best_model_name} and preprocessor saved as best_model.pkl and preprocessor.pkl")

# Load the best model and preprocessor
loaded_model = joblib.load('best_model.pkl')
loaded_preprocessor = joblib.load('preprocessor.pkl')
print(f"{best_model_name} and preprocessor loaded successfully")


# In[4]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')  # Update with the correct path

# Randomly pick 20 data points from the dataset
df_sample = df.sample(n=20, random_state=42)

# Define features and target variable
X_sample = df_sample.drop('selling_price', axis=1)  # Replace 'target_column' with the actual target column name
y_sample = df_sample['selling_price']

# Load the saved preprocessor and model
loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_model = joblib.load('best_model.pkl')
print("Model and preprocessor loaded successfully")

# Preprocess the sample data
X_sample_preprocessed = loaded_preprocessor.transform(X_sample)

# Apply the model to the subset and evaluate the performance
y_sample_pred = loaded_model.predict(X_sample_preprocessed)

# Calculate performance metrics
accuracy = accuracy_score(y_sample, y_sample_pred)
precision = precision_score(y_sample, y_sample_pred, average='weighted')
recall = recall_score(y_sample, y_sample_pred, average='weighted')
f1 = f1_score(y_sample, y_sample_pred, average='weighted')

try:
    roc_auc = roc_auc_score(y_sample, y_sample_pred, multi_class='ovr')
except ValueError:
    roc_auc = "ROC AUC score not applicable"

mse = mean_squared_error(y_sample, y_sample_pred)
mae = mean_absolute_error(y_sample, y_sample_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Confusion Matrix:\n{confusion_matrix(y_sample, y_sample_pred)}')

# Plot the confusion matrix
cm = confusion_matrix(y_sample, y_sample_pred)


# In[5]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# Define features and target variable
X = df.drop('km_driven', axis=1)  # Replace 'target_column' with the actual target column name
y = df['km_driven']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=10000)),
    'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
    'Random Forest': OneVsRestClassifier(RandomForestClassifier(random_state=42)),
    'Gradient Boosting': OneVsRestClassifier(GradientBoostingClassifier(random_state=42)),
    'Bagging': OneVsRestClassifier(BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)),
    'K-Nearest Neighbors': OneVsRestClassifier(KNeighborsClassifier())
}

# Define a function to evaluate models and store results
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = "ROC AUC score not applicable"

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}')
    print(f'{model_name} - Mean Squared Error (MSE): {mse}, Mean Absolute Error (MAE): {mae}')
    
    cm = confusion_matrix(y_test, y_pred)
   
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'MSE': mse,
        'MAE': mae
    }

# Train and evaluate models
results = []
for model_name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    result = evaluate_model(model, X_test_preprocessed, y_test, model_name)
    results.append(result)

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print(results_df)

# Find the best model based on accuracy
best_model_info = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nBest Model based on Accuracy:")
print(best_model_info)

# Save the best model and preprocessor
best_model_name = best_model_info['Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print(f"{best_model_name} and preprocessor saved as best_model.pkl and preprocessor.pkl")

# Load the best model and preprocessor
loaded_model = joblib.load('best_model.pkl')
loaded_preprocessor = joblib.load('preprocessor.pkl')
print(f"{best_model_name} and preprocessor loaded successfully")


# In[6]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')  # Update with the correct path

# Randomly pick 20 data points from the dataset
df_sample = df.sample(n=20, random_state=42)

# Define features and target variable
X_sample = df_sample.drop('km_driven', axis=1)  # Replace 'target_column' with the actual target column name
y_sample = df_sample['km_driven']

# Load the saved preprocessor and model
loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_model = joblib.load('best_model.pkl')
print("Model and preprocessor loaded successfully")

# Preprocess the sample data
X_sample_preprocessed = loaded_preprocessor.transform(X_sample)

# Apply the model to the subset and evaluate the performance
y_sample_pred = loaded_model.predict(X_sample_preprocessed)

# Calculate performance metrics
accuracy = accuracy_score(y_sample, y_sample_pred)
precision = precision_score(y_sample, y_sample_pred, average='weighted')
recall = recall_score(y_sample, y_sample_pred, average='weighted')
f1 = f1_score(y_sample, y_sample_pred, average='weighted')

try:
    roc_auc = roc_auc_score(y_sample, y_sample_pred, multi_class='ovr')
except ValueError:
    roc_auc = "ROC AUC score not applicable"

mse = mean_squared_error(y_sample, y_sample_pred)
mae = mean_absolute_error(y_sample, y_sample_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Confusion Matrix:\n{confusion_matrix(y_sample, y_sample_pred)}')

# Plot the confusion matrix
cm = confusion_matrix(y_sample, y_sample_pred)


# In[ ]:





# In[ ]:




