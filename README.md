# 🔍 Task 4: Logistic Regression Classifier – Breast Cancer Dataset
<br/>

## 📌 Project Overview

This project builds a **Binary Classification Model using Logistic Regression** on the **Breast Cancer Wisconsin Dataset**.  
The system classifies whether a tumor is **Benign (0)** or **Malignant (1)** using logistic regression and evaluates the model with several performance metrics.

<br/>

---

## ✨ Key Features
- ✅ Data loading and preprocessing (label encoding, feature standardization)
- ✅ Train-test split with `sklearn.model_selection`
- ✅ Logistic Regression model fitting
- ✅ Model evaluation using:
  - [Confusion Matrix](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/)
  - [Precision, Recall, F1-Score](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)
  - [ROC-AUC Score](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- ✅ [ROC Curve visualization](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html)
- ✅ Threshold tuning and performance impact demonstration
- ✅ [Sigmoid function](https://machinelearningmastery.com/a-gentle-introduction-to-sigmoid-function/) explanation with visualization

<br/>

---

## 🛠️ Tools & Libraries Used
- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/) – Data handling
- [Scikit-learn](https://scikit-learn.org/stable/) – Model building and evaluation
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – Visualizations

<br/>

---

## 📂 Dataset Description
- **[Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)**  
- Features: radius, texture, perimeter, area, smoothness, etc.
- Target: Diagnosis  
  - **M (Malignant)** → 1  
  - **B (Benign)** → 0

<br/>

---

## 🚀 Setup Instructions

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/Hanno-stud/task4-day4-Elevate-Labs.git
   ```

2. **Navigate to the Project Directory**  
   ```bash  
   cd task4-day4-Elevate-Labs
   ```

3. **Install Required Libraries**  
   ```bash  
   pip install pandas scikit-learn matplotlib seaborn  
   ```

4. **Run the Google Colab Python Notebook**  
   ```bash  
   Go to 'Google Colab'  >>  Copy 'Classification_with_Logistic_Regression.ipynb' file there...
   ```

5. **Run the Python Script (Optional)**  
   ```bash  
   python logistic_regression.py
   ```

<br/>

---

## 🔍 Project Workflow

### 1️⃣ Dataset Import & Preprocessing
- Loaded the dataset and dropped unnecessary columns.
- Encoded target variable: M → 1, B → 0.
- Standardized features using `StandardScaler`.

### 2️⃣ Train-Test Split
- Data split into **80% training and 20% testing sets.**

### 3️⃣ Logistic Regression Model
- Model trained using `LogisticRegression()` from scikit-learn.

### 4️⃣ Model Evaluation
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Score

### 5️⃣ Visualizations
- Confusion Matrix Plot
- ROC Curve
- Sigmoid Function Graph

### 6️⃣ Threshold Tuning
- Evaluated model performance at custom threshold (e.g., 0.3).

<br/>

---

## 📸 Real Outputs

### 1️⃣ Confusion Matrix
```text  
[[TN FP]  
 [FN TP]]  
``` 
![Confusion Matrix Output](https://github.com/user-attachments/assets/64a1f7b8-eb20-44e1-8c0f-d028492250bc)

---

### 2️⃣ Classification Report
```text  
               precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
``` 

---

### 3️⃣ ROC Curve
```text
ROC-AUC Score: 0.9974
```
![ROC Curve Output](https://github.com/user-attachments/assets/0a4fc446-b9a8-4df7-868e-2479eeb32574)


---

### 4️⃣ Confusion Matrix at Custom Threshold 0.3
```text  
 [[67  4]
 [ 1 42]]
``` 

---

### 5️⃣ Sigmoid Function Curve
![Sigmoid Function Plot](https://github.com/user-attachments/assets/5ce092f6-76d3-4f87-be98-75085b65e8a3)

<br/>

---

## 🔮 Future Scope
- Implement grid search for hyperparameter tuning.
- Add multi-class logistic regression with other datasets.
- Explore advanced model evaluation with cross-validation.
- Deploy the model using Flask or FastAPI.

<br/>

---

## 🙋‍♂️ Author
**IVIN SANTHOSH**  
Python Developer | Machine Learning Enthusiast

<br/>

---

## 🙏 Acknowledgments
- Thanks to Scikit-learn, Matplotlib, and Seaborn documentation teams.
- Breast Cancer Wisconsin Dataset provided via Kaggle.

<br/>

---
