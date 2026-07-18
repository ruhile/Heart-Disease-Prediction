# ❤️ Heart Disease Prediction

An end-to-end Machine Learning project to predict the presence of heart disease using clinical patient measurements from the combined UCI Heart Disease databases.

## 📝 Overview
Cardiovascular diseases are the leading cause of death globally. Early detection of heart disease plays a vital role in medical diagnostics and patient care. This project implements a machine learning classification pipeline to predict whether a patient has heart disease based on common clinical measurements such as age, cholesterol, chest pain type, resting blood pressure, and maximum heart rate achieved.

The codebase implements a complete data science workflow beginning with raw data exploration and cleaning, moving to categorical encoding, stratification train-test partitioning, and model training. We implement and compare three candidate classifiers: Logistic Regression, Decision Tree, and Random Forest, evaluating them on key diagnostic metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Ultimately, the best performing model is serialized and packaged with an inference script to serve predictions on new, unseen clinical patient records.

---

## 📊 Dataset
This project uses the combined **UCI Heart Disease Dataset** containing **920 patient records** from Cleveland, Hungary, Switzerland, and Long Beach VA databases.

- **Number of Records:** 920 patients
- **Number of Features:** 13 clinical attributes + 1 binary target column

---

## 🛠️ Technologies Used
- **Language:** Python 3.12
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Exploration Environment:** Jupyter Notebook

---

## ⚙️ Workflow
```text
  Dataset
     ↓
Data Cleaning
     ↓
    EDA
     ↓
Preprocessing
     ↓
Logistic Regression
     ↓
Decision Tree
     ↓
Random Forest
     ↓
Model Evaluation
     ↓
Save Best Model
```

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── data/
│   ├── heart_disease_uci.csv   # Combined raw UCI dataset
│   ├── heart_clean_day1.csv    # Cleaned dataset (unnecessary columns dropped)
│   ├── heart_processed.csv     # Preprocessed dataset (imputed & encoded)
│   └── model_comparison.csv    # Final tabular model metrics comparison CSV
│
├── notebooks/
│   └── heart_disease_eda.ipynb # Jupyter notebook for Exploratory Data Analysis & Modeling
│
├── src/
│   └── predict.py              # Inference script for patient classification
│
├── models/
│   └── heart_disease_model.pkl # Saved trained Random Forest model weights (Git-ignored)
│
├── images/
│   ├── correlation_heatmap.png # Feature correlation matrix heatmap
│   ├── confusion_matrix.png    # Random Forest evaluation confusion matrix
│   ├── decision_tree.png       # Visualized Decision Tree conditional splits
│   ├── feature_importance.png  # Random Forest Mean Decrease in Impurity features importance
│   ├── model_comparison.png    # Bar chart comparing all models side-by-side
│   └── roc_curve.png           # Plotted ROC comparison curves (AUC)
│
├── README.md                   # Project documentation & reflections
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules
└── LICENSE                     # MIT License
```

---

## 📈 Results

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **82.07%** | **83.50%** | 84.31% | 83.90% | **0.8922** |
| **Decision Tree** | **76.09%** | 77.36% | 80.39% | 78.85% | **0.7556** |
| **Random Forest** | **82.61%** | 82.41% | **87.25%** | **84.76%** | **0.9075** |

- *The Random Forest Classifier (100 Trees) achieved the best overall performance, capturing **87.25% of positive cases** (highest recall) with an excellent AUC of **0.91**.*

---

## 🚀 Future Improvements
- **Hyperparameter Tuning:** Conduct a systematic Grid Search or Random Search over Random Forest attributes (e.g. `max_depth`, `min_samples_split`, `max_features`) to further boost accuracy.
- **Advanced Classifiers:** Train and compare gradient boosted trees (e.g., XGBoost, LightGBM) to see if boosting yields marginal accuracy gains.
- **Cross-Validation:** Implement Stratified K-Fold cross-validation to guarantee model stability and minimize validation variance.
- **Web Application:** Build an interactive frontend using Streamlit or a Flask/FastAPI REST backend to deploy the model for live patient predictions.

---

## 📝 Final Reflection

### 1. Which model performed best?
The **Random Forest Classifier (100 Trees)** performed best, achieving the highest overall test accuracy (**82.61%**), highest test F1-score (**84.76%**), and the highest Area Under the ROC Curve (**0.9075**).

### 2. Why did you choose that model?
We selected Random Forest because it offers the most stable generalization. It uses bagging and random feature selection to build an ensemble of diverse trees, which cancels out individual tree variance, resulting in a higher test score than a single Decision Tree (76.09%) and better class separation capacity than Logistic Regression (AUC 0.91 vs. 0.89).

### 3. Which evaluation metric mattered most for this healthcare problem?
**Recall (Sensitivity)** was the most critical evaluation metric. Recall measures the model's ability to identify all actual positive heart disease cases. In medicine, a False Negative (misclassifying a sick patient as healthy) is highly dangerous and can lead to untreated illness or death, whereas a False Positive (identifying a healthy patient as diseased) will be flagged and corrected during subsequent clinical tests.

### 4. What preprocessing steps improved performance?
- **Median Imputation** for numeric variables: This was robust against outliers and clinical zero-codes in features like cholesterol.
- **Categorical Mode Imputation & Label Encoding**: Allowed the mathematical scikit-learn models to process string variables like chest pain and thalassemia correctly.
- **Feature Scaling**: Scaled numeric features to unit variance, which is essential to prevent scale-bias and help gradient solvers converge in the Logistic Regression model.

### 5. What would you improve if you continued this project?
- Implement **K-Fold cross-validation** to get a more robust estimation of test accuracy.
- Use **SMOTE** (Synthetic Minority Over-sampling Technique) or other sampling options if target class imbalances occur.
- Package the serialized model weights into a **Streamlit Web application** to let clinical operators input readings directly through a browser UI.
