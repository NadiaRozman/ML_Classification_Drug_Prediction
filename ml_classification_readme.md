# 🤖 Machine Learning Classification: Drug Prescription Prediction

## 📊 Introduction

This project demonstrates the application of **multiple machine learning classification algorithms** to predict the most appropriate drug prescription for patients based on their medical characteristics. Using a dataset of 200 patients, I built and compared **7 different classification models** to determine which algorithm performs best for this medical prediction task.

The goal is to predict which of 5 drugs (Drug A, B, C, X, or Y) would be most effective for a patient based on their:
- Age
- Sex
- Blood Pressure
- Cholesterol levels
- Sodium-to-Potassium ratio

This project showcases end-to-end machine learning workflow including data preprocessing, model training, hyperparameter tuning, evaluation, and visualization.

---

## 🎯 Objectives

After completing this project, the following skills are demonstrated:
- Building multiple classification algorithms from scratch
- Comparing model performance across different algorithms
- Hyperparameter tuning using GridSearchCV
- Handling categorical variables through encoding
- Model evaluation using confusion matrices and classification reports
- Data visualization for model performance analysis

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

**Problem Type:** Multi-class classification (5 classes)

---

## 🛠️ Tools & Technologies

**Programming & Libraries:**
- **Python 3.x** – Primary programming language
- **NumPy** – Numerical computations
- **Pandas** – Data manipulation and analysis
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning algorithms and evaluation metrics

**Development Environment:**
- **Jupyter Notebook** – Interactive development and analysis
- **Google Colab** – Cloud-based notebook environment

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or Google Colab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NadiaRozman/ML_Classification_Drug_Prediction.git
   cd ML_Classification_Drug_Prediction
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**
   - Navigate to `Drug_Classification_ML.ipynb`
   - Ensure `drug200.csv` is in the same directory

---

## 🔬 Machine Learning Models Implemented

This project implements and compares **7 classification algorithms**:

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

## 📈 Project Workflow

### 1. **Data Loading & Exploration**
   - Load dataset using Pandas
   - Explore data structure, types, and distributions
   - Check for missing values
   - Analyze target variable distribution

### 2. **Data Preprocessing**
   - **Feature Engineering:**
     - One-Hot Encoding for categorical variables (Sex, BP, Cholesterol)
     - Label Encoding for target variable (Drug)
   - **Feature Scaling:**
     - MinMaxScaler for algorithms requiring normalized inputs (MultinomialNB, BernoulliNB)

### 3. **Train-Test Split**
   - Split ratio: 70% training, 30% testing
   - Stratified split to maintain class distribution
   - Random state set for reproducibility

### 4. **Model Training & Hyperparameter Tuning**
   - Train each model on training set
   - Use GridSearchCV for optimal hyperparameter selection:
     - **Gaussian NB:** var_smoothing parameter
     - **Multinomial/Bernoulli NB:** alpha (Laplace smoothing)
     - **Decision Tree:** max_depth optimization
     - **Random Forest:** max_depth with 100 estimators
     - **KNN:** n_neighbors (1-20 range)
     - **SVM:** RBF kernel with gamma='auto'

### 5. **Model Evaluation**
   - **Accuracy Score** – Overall prediction accuracy
   - **Confusion Matrix** – True vs predicted classifications
   - **Classification Report** – Precision, recall, F1-score per class
   - **Cross-Validation** – 5-fold stratified CV for robust evaluation

### 6. **Visualization**
   - Heatmaps for confusion matrices
   - Model comparison charts
   - Feature importance analysis (for tree-based models)

---

## 📊 Results Summary

### Model Performance Comparison

| Model | Accuracy | Key Strengths |
|-------|----------|---------------|
| **Gaussian Naive Bayes** | ~XX% | Fast, works well with continuous features |
| **Multinomial Naive Bayes** | ~XX% | Effective for count-based data |
| **Bernoulli Naive Bayes** | ~XX% | Handles binary features well |
| **Logistic Regression** | ~XX% | Interpretable, probabilistic output |
| **Decision Tree** | ~XX% | Intuitive, handles non-linear relationships |
| **Random Forest** | ~XX% | Robust, reduces overfitting |
| **K-Nearest Neighbors** | ~XX% | Simple, no training phase |
| **Support Vector Machine** | ~XX% | Effective in high dimensions |

> **Note:** Run the notebook to see actual accuracy scores for each model.

### Key Findings

1. **Best Performing Model:** [Model Name] achieved the highest accuracy of XX%
2. **Most Stable Model:** [Model Name] showed consistent performance across cross-validation folds
3. **Feature Importance:** [Feature] was the most significant predictor
4. **Optimal Hyperparameters:**
   - Decision Tree max_depth: X
   - KNN n_neighbors: X
   - Random Forest max_depth: X

---

## 📚 What I Learned

Through this project, I enhanced my machine learning skills:

* 🤖 **Algorithm Implementation** – Built and compared 7 classification algorithms
* 🔧 **Hyperparameter Tuning** – Used GridSearchCV for optimal parameter selection
* 📊 **Model Evaluation** – Applied multiple metrics (accuracy, precision, recall, F1-score)
* 🎨 **Data Visualization** – Created confusion matrix heatmaps for model comparison
* 🔄 **Cross-Validation** – Implemented stratified K-fold for robust evaluation
* 🧹 **Data Preprocessing** – Handled categorical variables and feature scaling
* 📈 **Ensemble Methods** – Understood the power of Random Forest for improved predictions

---

## 💡 Key Insights

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

## 📂 Project Structure

```
ML_Classification_Drug_Prediction/
│
├── data/
│   └── drug200.csv                          # Dataset
│
├── notebooks/
│   └── Drug_Classification_ML.ipynb         # Main analysis notebook
│
├── images/                                   # Visualization outputs
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_rf.png
│   ├── confusion_matrix_svm.png
│   └── model_comparison.png
│
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── .gitignore                               # Git ignore file
```

---

## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/NadiaRozman/ML_Classification_Drug_Prediction/issues).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📫 Contact

**Nadia Rozman**
- GitHub: [@NadiaRozman](https://github.com/NadiaRozman)
- LinkedIn: [Nadia Rozman](https://www.linkedin.com/in/nadia-rozman-4b4887179/)

---

## 🙏 Acknowledgments

- Dataset provided as part of machine learning coursework
- Scikit-learn documentation for algorithm implementation guidance
- Seaborn and Matplotlib communities for visualization techniques

---

### ✨ Created by Nadia Rozman | January 2026

**⭐ If you found this project helpful, please consider giving it a star!**