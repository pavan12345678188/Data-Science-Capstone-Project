#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# Define features and target variable
X = df.drop('year', axis=1)  
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

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = "ROC AUC score not applicable"

    cm = confusion_matrix(y_test, y_pred)

    print(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, MSE: {mse}, MAE: {mae}, ROC AUC: {roc_auc}')
    print(f'{model_name} - Confusion Matrix:\n{cm}\n')
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MSE': mse,
        'MAE': mae,
        'ROC AUC': roc_auc
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


# In[8]:


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


# In[10]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv') 

# Randomly pick 20 data points from the dataset
df_sample = df.sample(n=20, random_state=42)
df_sample

# Define features and target variable
X_sample = df_sample.drop('year', axis=1)  
y_sample = df_sample['year']

# Load the saved preprocessor and model
loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_model = joblib.load('best_model.pkl')
print("Model and preprocessor loaded successfully")

# Preprocess the sample data
X_sample_preprocessed = loaded_preprocessor.transform(X_sample)

# Apply the model to the subset and evaluate the performance
predictions = loaded_model.predict(X_sample_preprocessed)

# Calculate performance metrics
accuracy = accuracy_score(y_sample,predictions)
precision = precision_score(y_sample,predictions, average='weighted')
recall = recall_score(y_sample,predictions, average='weighted')
f1 = f1_score(y_sample, predictions, average='weighted')

try:
    roc_auc = roc_auc_score(y_sample, predictions, multi_class='ovr')
except ValueError:
    roc_auc = "ROC AUC score not applicable"

mse = mean_squared_error(y_sample,predictions)
mae = mean_absolute_error(y_sample,predictions )



print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Predictions:\n{(y_sample, predictions)}')


# In[11]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# Define features and target variable
X = df.drop('selling_price', axis=1)  
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

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = "ROC AUC score not applicable"

    cm = confusion_matrix(y_test, y_pred)

    print(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, MSE: {mse}, MAE: {mae}, ROC AUC: {roc_auc}')
    print(f'{model_name} - Confusion Matrix:\n{cm}\n')
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MSE': mse,
        'MAE': mae,
        'ROC AUC': roc_auc
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


# In[12]:


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


# In[13]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv') 

# Randomly pick 20 data points from the dataset
df_sample = df.sample(n=20, random_state=42)
df_sample

# Define features and target variable
X_sample = df_sample.drop('selling_price', axis=1)  
y_sample = df_sample['selling_price']

# Load the saved preprocessor and model
loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_model = joblib.load('best_model.pkl')
print("Model and preprocessor loaded successfully")

# Preprocess the sample data
X_sample_preprocessed = loaded_preprocessor.transform(X_sample)

# Apply the model to the subset and evaluate the performance
predictions = loaded_model.predict(X_sample_preprocessed)

# Calculate performance metrics
accuracy = accuracy_score(y_sample,predictions)
precision = precision_score(y_sample,predictions, average='weighted')
recall = recall_score(y_sample,predictions, average='weighted')
f1 = f1_score(y_sample, predictions, average='weighted')

try:
    roc_auc = roc_auc_score(y_sample, predictions, multi_class='ovr')
except ValueError:
    roc_auc = "ROC AUC score not applicable"

mse = mean_squared_error(y_sample,predictions)
mae = mean_absolute_error(y_sample,predictions )



print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Predictions:\n{(y_sample, predictions)}')


# In[14]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# Define features and target variable
X = df.drop('km_driven', axis=1)  
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

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = "ROC AUC score not applicable"

    cm = confusion_matrix(y_test, y_pred)

    print(f'{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, MSE: {mse}, MAE: {mae}, ROC AUC: {roc_auc}')
    print(f'{model_name} - Confusion Matrix:\n{cm}\n')
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MSE': mse,
        'MAE': mae,
        'ROC AUC': roc_auc
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


# In[15]:


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


# In[16]:


# Load the dataset
df = pd.read_csv('CAR DETAILS.csv') 

# Randomly pick 20 data points from the dataset
df_sample = df.sample(n=20, random_state=42)
df_sample

# Define features and target variable
X_sample = df_sample.drop('km_driven', axis=1)  
y_sample = df_sample['km_driven']

# Load the saved preprocessor and model
loaded_preprocessor = joblib.load('preprocessor.pkl')
loaded_model = joblib.load('best_model.pkl')
print("Model and preprocessor loaded successfully")

# Preprocess the sample data
X_sample_preprocessed = loaded_preprocessor.transform(X_sample)

# Apply the model to the subset and evaluate the performance
predictions = loaded_model.predict(X_sample_preprocessed)

# Calculate performance metrics
accuracy = accuracy_score(y_sample,predictions)
precision = precision_score(y_sample,predictions, average='weighted')
recall = recall_score(y_sample,predictions, average='weighted')
f1 = f1_score(y_sample, predictions, average='weighted')

try:
    roc_auc = roc_auc_score(y_sample, predictions, multi_class='ovr')
except ValueError:
    roc_auc = "ROC AUC score not applicable"

mse = mean_squared_error(y_sample,predictions)
mae = mean_absolute_error(y_sample,predictions )



print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Predictions:\n{(y_sample, predictions)}')


# In[ ]:





# In[ ]:




