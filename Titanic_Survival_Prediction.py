# ==============================================================================
# Titanic Survival Prediction using Random Forest
# Author: [Your Name]
# Description: End-to-End Machine Learning Pipeline (Data Cleaning -> Modeling -> Deployment)
# ==============================================================================

# 1. Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # For saving the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
print("Loading data...")
df = sns.load_dataset('titanic')

# ==============================================================================
# 3. Data Cleaning & Feature Engineering
# ==============================================================================

# A. Smart Imputation for 'Age'
# Instead of using the global mean, we fill missing ages based on 'Pclass' and 'Sex' groups.
# This preserves the data distribution better.
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.mean()))

# B. Fill missing 'Embarked' values with the mode (most common value)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# C. Fill missing 'Fare' (if any) with the median
df['fare'] = df['fare'].fillna(df['fare'].median())

# D. Create New Feature: 'Family_Size'
df['family_size'] = df['sibsp'] + df['parch']

# E. Create New Feature: 'Is_Alone'
# 1 if the passenger is traveling alone, 0 otherwise
df['is_alone'] = (df['family_size'] == 0).astype(int)

# ==============================================================================
# 4. Data Encoding & Preprocessing
# ==============================================================================

# Convert categorical variables ('sex', 'embarked') into dummy/indicator variables (One-Hot Encoding)
# drop_first=True is used to avoid multicollinearity (dummy variable trap)
df_encoded = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Define Features (X) and Target (y)
feature_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'is_alone', 'sex_male', 'embarked_Q', 'embarked_S']
X = df_encoded[feature_cols]
y = df_encoded['survived']

# Split data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data ready! Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# ==============================================================================
# 5. Model Training (Random Forest)
# ==============================================================================

# Initialize Random Forest Classifier
# n_estimators=100: Number of trees in the forest
# max_depth=10: Limit depth to prevent overfitting (Hyperparameter Tuning)
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
print("Training the Random Forest model...")
model_rf.fit(X_train, y_train)

# ==============================================================================
# 6. Evaluation
# ==============================================================================

# Make predictions on the test set
predictions = model_rf.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)
print("-" * 30)
print(f"✅ Final Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

# Show detailed classification report
print("Classification Report:\n")
print(classification_report(y_test, predictions))

# ==============================================================================
# 7. Feature Importance Visualization
# ==============================================================================

# Extract feature importances
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()

# Save the plot for LinkedIn/GitHub
plt.savefig('feature_importance.png', dpi=300)
print("📊 Feature importance plot saved as 'feature_importance.png'")
plt.show()

# ==============================================================================
# 8. Model Deployment (Saving)
# ==============================================================================

# Save the trained model to a file
model_filename = 'titanic_rf_model.pkl'
joblib.dump(model_rf, model_filename)

print(f"💾 Model saved successfully as '{model_filename}'")
print("Ready for deployment!")