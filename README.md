# 🩺 Heart Disease Prediction Model

An end-to-end Machine Learning project to predict the presence and severity of heart disease using clinical patient measurements from the combined UCI Heart Disease databases.

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── data/
│   ├── heart_disease_uci.csv   # Combined UCI Heart Disease dataset (920 patients)
│   ├── heart_clean_day1.csv    # Day 1 cleaned dataset (unnecessary columns dropped)
│   ├── heart_processed.csv     # Day 2 preprocessed dataset (imputed & encoded)
│   ├── X_train.npy             # Scaled training features (NumPy array)
│   ├── X_test.npy              # Scaled testing features (NumPy array)
│   ├── y_train.npy             # Training target labels (NumPy array)
│   └── y_test.npy              # Testing target labels (NumPy array)
│
├── notebooks/
│   ├── heart_disease_eda.ipynb           # Jupyter notebook for Exploratory Data Analysis (Day 1)
│   └── heart_disease_preprocessing.ipynb # Jupyter notebook for Data Preprocessing (Day 2)
│
├── src/                        # Source code directory for programmatic steps
│   └── preprocess.py           # Preprocessing and scaling module pipeline
│
├── images/                     # Saved plots and visual assets
│
├── README.md                   # Project documentation
│
├── requirements.txt            # Python package dependencies
│
└── .gitignore                  # Git ignore file
```

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have **Python 3.8+** installed.

### 2. Install Dependencies
Run the following command to install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Running EDA & Preprocessing
To run the notebooks:
```bash
# To run EDA
jupyter notebook notebooks/heart_disease_eda.ipynb

# To run Preprocessing
jupyter notebook notebooks/heart_disease_preprocessing.ipynb
```

To execute the preprocessing programmatically:
```bash
python src/preprocess.py
```

---

## 📊 Dataset Attribute Information

The combined dataset contains **920 patient records** from Cleveland, Hungary, Switzerland, and Long Beach VA databases, featuring the following attributes:

| Column Name | Type | Description |
|---|---|---|
| `id` | Integer | Unique identifier |
| `age` | Integer | Age in years |
| `sex` | Categorical | Sex (Male, Female) |
| `dataset` | Categorical | Origin database (Cleveland, Hungary, Switzerland, Long Beach VA) |
| `cp` | Categorical | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic) |
| `trestbps` | Numeric | Resting blood pressure (in mm Hg on admission to the hospital) |
| `chol` | Numeric | Serum cholestoral in mg/dl |
| `fbs` | Boolean | Fasting blood sugar > 120 mg/dl (TRUE, FALSE) |
| `restecg` | Categorical | Resting electrocardiographic results (normal, st-t abnormality, lv hypertrophy) |
| `thalach` | Numeric | Maximum heart rate achieved |
| `exang` | Boolean | Exercise induced angina (TRUE, FALSE) |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | The slope of the peak exercise ST segment (upsloping, flat, downsloping) |
| `ca` | Numeric | Number of major vessels (0-3) colored by fluoroscopy |
| `thal` | Categorical | Thalassemia type (normal, fixed defect, reversable defect) |
| **`num`** (Target) | Integer | Diagnosis of heart disease (0 = normal, 1-4 = presence of heart disease) |
