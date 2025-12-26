# 🚢 Titanic Survival Prediction: From Logistics to Random Forest

This project analyzes the famous Titanic dataset to predict passenger survival. It demonstrates the transition from a simple baseline model to a robust **Random Forest Classifier**, achieving an accuracy of **~84%**.

## 🚀 Key Highlights
- **Data Engineering:** - Imputed missing 'Age' values based on Pclass and Sex groups (Smart Imputation).
  - Created new features like `Family_Size` and `Is_Alone`.
  - Engineered `Title` extraction from names.
- **Model Evolution:**
  - Started with **Logistic Regression** (~80% Accuracy).
  - Upgraded to **Random Forest Classifier** with Hyperparameter Tuning (`max_depth=10`).
  - Achieved a final accuracy of **83.8%**.
- **Insight Discovery:**
  - Unlike linear models, the Random Forest revealed that **Ticket Price (Fare)** contains more granular information than Class, ranking it as the 2nd most important feature after Gender.

## 🛠️ Tech Stack
- Python
- Pandas & NumPy (Vectorized operations)
- Scikit-learn (Random Forest, Logistic Regression)
- Seaborn & Matplotlib (EDA)

## 📊 Results
The final model is saved as `my_titanic_model.pkl` and is ready for deployment.
