# 🤖 Machine Learning Classification: Drug Prescription Prediction

## 📊 Introduction

This project implements and compares multiple **machine learning classification algorithms** to predict the most appropriate drug prescription for patients based on their medical characteristics.

The goal is to identify which ML models perform best for this **multi-class classification problem** and provide actionable insights into **accuracy and F1 performance** for each algorithm.

**Models Implemented:**

1. Gaussian Naive Bayes
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes
4. Logistic Regression
5. Decision Tree
6. Random Forest
7. K-Nearest Neighbors (KNN)
8. Support Vector Machine (SVM)

---

## 🎯 Background
The inspiration for this project came from a desire to **compare classical ML classifiers** on a healthcare-inspired dataset and explore how different algorithms handle **categorical and numerical patient data**.

Key questions explored:

- Which classification algorithm achieves the **highest accuracy** for drug prediction?

- How do models compare in terms of **F1 Macro score**, especially for minority drug classes?

- What insights can we gain from the **confusion matrices** for each model?

- The dataset is **synthetic and educational**, containing 200 patient records with features such as age, sex, blood pressure, cholesterol, and Na_to_K ratio.

> ⚠️ **Disclaimer**: This dataset does **not represent real patients or medical advice**. The project is purely for **learning purposes**.

---

## 🗂️ Dataset Overview

**Source:** Drug prescription dataset with 200 patient records

**Features:**
- **Age** (numeric): Patient's age
- **Sex** (categorical): M (Male) or F (Female)
- **BP** (categorical): Blood Pressure - HIGH, NORMAL, or LOW
- **Cholesterol** (categorical): HIGH or NORMAL
- **Na_to_K** (numeric): Sodium to Potassium ratio in blood

**Target Variable:**
- **Drug** (categorical): Drug A, Drug B, Drug C, Drug X, or Drug Y

**Problem Type:** Multi-class classification (8 classes)

---

## 🛠️ Tools I Used

**Programming & Libraries:**
- **Python** – Analysis and model implementation
- **NumPy** – Numerical computations
- **Pandas** – Data manipulation and analysis
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning models, preprocessing, metrics and model evaluation

**Development Environment:**
- **Jupyter Notebook** – Interactive development and analysis
- **Google Colab** – Cloud-based notebook environment

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- Jupyter Notebook or Google Colab
- Conda (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NadiaRozman/ML_Classification_Drug_Prediction.git
   cd ML_Classification_Drug_Prediction
   ```

2. **Set up the environment**

   - **Option 1: Using Conda**
   ```bash
   conda env create -f environment.yml
   conda activate ml_drug_prediction
   ```

   - **Option 2: Using pip**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open and run the notebook**
   - Navigate to `Drug_Classification_ML.ipynb`
   - Ensure `drug_prescription_dataset.csv` is in the same directory

---

## 🔬 Machine Learning Models Implemented

This project implements and compares **8 classification algorithms**:

### 1. **Naive Bayes Family**
   - **Gaussian Naive Bayes** – Assumes continuous features with Gaussian distribution
   - **Multinomial Naive Bayes** – Suitable for discrete count data
   - **Bernoulli Naive Bayes** – Binary/boolean features

### 2. **Logistic Regression**
   - Linear model for classification with probabilistic output
   - Used with stratified cross-validation

### 3. **Decision Tree**
   - Tree-based model using entropy criterion
   - Includes hyperparameter tuning for max_depth

### 4. **Random Forest**
   - Ensemble method combining multiple decision trees
   - 100 estimators with optimized max_depth

### 5. **K-Nearest Neighbors (KNN)**
   - Instance-based learning algorithm
   - Optimized number of neighbors through GridSearchCV

### 6. **Support Vector Machine (SVM)**
   - Uses RBF kernel for non-linear classification
   - Effective for high-dimensional data

---

## 📈 Project Workflow & Analysis

### 1. **Data Loading & Exploration**
   - Load dataset using Pandas
   - Explore data structure, types, and distributions
   - Check for missing values
   - Analyze target variable distribution

### 2. **Data Preprocessing**
   - One-hot encoded categorical variables (Sex, BP, Cholesterol)
   - Label encoded the target variable Drug

### 3. **Train-Test Split**
   - Split ratio: 70% training, 30% testing
   - Stratified split to maintain class distribution
   - Random state set for reproducibility

### 4. **Model Evaluation**
   - **Metrics**: Accuracy, Macro F1, Precision, Recall; Confusion matrices visualize per-class performance.
   - 5-fold Stratified cross-validation for robust evaluation

### 5. **Test Set Performance Summary**

| Model                  | Accuracy (Test) | F1 Macro (Test) |
| ---------------------- | --------------- | --------------- |
| Decision Tree          | 0.9833          | 0.9644          |
| Random Forest          | 0.9667          | 0.9250          |
| Logistic Regression    | 0.8333          | 0.8124          |
| Support Vector Machine | 0.8333          | 0.8124          |
| Gaussian NB            | 0.7833          | 0.7752          |
| K-Nearest Neighbors    | 0.7667          | 0.7476          |
| Bernoulli NB           | 0.5000          | 0.3343          |
| Multinomial NB         | 0.5833          | 0.2735          |

> **Note:** Run the notebook to see actual accuracy scores for each model.

**Visualization – Test Set Accuracy & F1 Macro**

![Test Set Performance Summary](images/Test_Set_Accuracy_Comparison.png)
*Accuracy and F1 Macro comparison across all models.*

---

### Key Findings

1. **Best Performing Model:** **Decision Tree** achieved the highest accuracy of **98.33%** on the test set.
2. **Most Stable Model:** **Random Forest** showed consistent performance across cross-validation folds, F1 Macro **0.925**.
3. **Feature Importance:** **Na_to_K** was the most significant predictor.
4. **Optimal Hyperparameters:**
   - Decision Tree `max_depth`: 4
   - K-Nearest Neighbors `n_neighbors`: 18
   - Random Forest `max_depth`: 4

> Tree-based models perform extremely well on this dataset, while feature scaling enables KNN and Logistic Regression to achieve competitive performance.

---

## 📚 What I Learned

Through this project, I enhanced my machine learning and Python skills:

* 🤖 **Algorithm Implementation** – Built and compared 8 classification algorithms
* **💡 Pipeline Design** – Unified pipelines enabled scaling, training, and evaluation with minimal code duplication
* 🔧 **Hyperparameter Tuning** – Used GridSearchCV and manual tuning for optimal parameters
* 📊 **Model Evaluation** – Applied multiple metrics (accuracy, precision, recall, F1-score). Accuracy alone is insufficient for imbalanced multi-class problems; F1 Macro provides a more balanced view
* 🎨 **Data Visualization** – Annotated bar charts and heatmaps improved interpretability and presentation quality
* 🔄 **Cross-Validation** – Implemented stratified K-fold for robust evaluation
* 🧹 **Data Preprocessing** – Handled categorical variables, feature scaling, and label encoding
* 📈 **Ensemble Methods** – Understood the power of Random Forest for improved predictions
* 🔑 **Key Insights:**
   - Decision Tree = Best performing model
   - Random Forest = Most stable model
   - Na_to_K = Most important feature
* 🐍 **Python & Scikit-learn** – Reinforced expertise in preprocessing, model tuning, cross-validation, and classification workflows

---

### 💡 Key Insights

1. **Ensemble methods generally outperform single models** – Random Forest showed superior stability
2. **Proper encoding is critical** – One-hot encoding significantly improved model performance
3. **Hyperparameter tuning matters** – GridSearchCV improved accuracy by 5-10% for most models
4. **Class imbalance affects performance** – Stratified splitting ensures representative train/test sets
5. **Different algorithms suit different scenarios:**
   - Naive Bayes: Fast inference for real-time predictions
   - Decision Trees: Interpretable for medical decisions
   - SVM: Effective when feature space is complex

---

## 🔮 Future Enhancements

Potential improvements for this project:

1. **Feature Engineering:**
   - Create interaction terms (e.g., Age × Na_to_K ratio)
   - Polynomial features for non-linear relationships

2. **Advanced Models:**
   - XGBoost or LightGBM for gradient boosting
   - Neural Networks for complex pattern recognition

3. **Model Interpretability:**
   - SHAP values for feature importance
   - LIME for local explanations

4. **Deployment:**
   - Flask API for model serving
   - Streamlit app for interactive predictions

5. **Extended Analysis:**
   - ROC curves and AUC scores
   - Learning curves to detect overfitting
   - Feature selection using RFECV

---

### ✨ Created by Nadia Rozman | January 2026

**📂 Project Structure**

```
ML_Classification_Drug_Prediction/
│
├── data/
│   └── drug_prescription_dataset.csv
│
├── notebooks/
│   └── Drug_Classification_ML.ipynb
│
├── images/
│   └── Test_Set_Accuracy_Comparison.png
│
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

**🔗 Connect with me**
- GitHub: [@NadiaRozman](https://github.com/NadiaRozman)
- LinkedIn: [Nadia Rozman](https://www.linkedin.com/in/nadia-rozman-4b4887179/)

**⭐ If you found this project helpful, please consider giving it a star!**
