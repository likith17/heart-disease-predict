# Heart Disease Prediction using Machine Learning

This repository contains an end-to-end Machine Learning project focused on predicting **heart disease** using structured patient data.  
The project includes **data preprocessing, feature selection, model building, evaluation, PCA analysis, and deep learning models**.

---

## Project Overview

Heart disease is one of the leading causes of death globally. Early prediction can help save lives by enabling timely intervention.  
This project builds a complete ML pipeline to classify whether a person is at risk of heart disease using various clinical attributes.

The workflow includes:

- Loading and preprocessing the dataset  
- Feature selection using **SelectKBest**  
- Training multiple ML models  
- Performance comparison  
- Visual evaluation (confusion matrices, PCA plots)  
- Deep learning model using **TensorFlow/Keras**

---

## Dataset

The project uses the standard **Heart Disease dataset** (commonly from UCI repository or Kaggle).  
Key attributes typically include:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol  
- Fasting blood sugar  
- Resting ECG results  
- Maximum heart rate  
- Exercise-induced angina  
- ST depression  
- Major vessels  
- Thalassemia  
- **Target** (1 = disease, 0 = healthy)

> The dataset is loaded as `heart_df` inside the notebook.

---

## Data Preprocessing

The notebook performs:

- Missing value checks  
- Train–test split (80/20)  
- Feature scaling using **StandardScaler**  
- Feature selection using **SelectKBest (ANOVA F-test)**  
- Extraction of the top 5 most correlated features  

---

## 🤖 Machine Learning Models Used

The following ML algorithms were trained and evaluated:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier |
| **Decision Tree Classifier** | Rule-based non-linear classifier |
| **Random Forest Classifier** | Ensemble of decision trees |
| **Gradient Boosting Classifier** | Boosted trees for improved accuracy |
| **Support Vector Classifier (SVC)** | Margin-based non-linear classifier |
| **MLPClassifier** | Neural network (sklearn) |

---

## Model Evaluation

For each model, the notebook computes:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix visualization  


