# =============================================================================
# SETUP: Create images directory if it doesn't exist
# =============================================================================
import os

# Create images directory if it doesn't exist
if not os.path.exists('../images'):
    os.makedirs('../images')
    print("Created 'images' directory")
else:
    print("'images' directory already exists")

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# =============================================================================
# LOAD AND EXPLORE DATA
# =============================================================================
# Load the dataset
my_data = pd.read_csv('drug200.csv')

# Display first few rows
print("Dataset Preview:")
print(my_data.head())

# Check the shape
print(f"\nDataset Shape: {my_data.shape}")

# Check data types
print("\nData Types:")
print(my_data.dtypes)

# Get statistical description
print("\nStatistical Description:")
print(my_data.describe())

# Check for missing values
print("\nMissing Values:")
print(my_data.isnull().sum())

# Check target distribution
print("\nTarget Variable Distribution:")
print(my_data['Drug'].value_counts())

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
# One-Hot Encoding for categorical variables
X_encoded = pd.get_dummies(my_data,
                          columns=['Sex', 'BP', 'Cholesterol'],
                          drop_first=True)

# Label encode the target variable
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(my_data['Drug'])

# X = features (everything except the target)
X = X_encoded.drop('Drug', axis=1)
Y = Y_encoded

print("\nEncoded Features Shape:", X.shape)
print("Encoded Target Shape:", Y.shape)

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================
# Split the data with stratification
X_trainset, X_testset, Y_trainset, Y_testset = train_test_split(
    X, Y, test_size=0.3, random_state=3, stratify=Y
)

print("\nTraining Set Shapes:")
print(f"X_trainset shape: {X_trainset.shape}")
print(f"Y_trainset shape: {Y_trainset.shape}")

print("\nTesting Set Shapes:")
print(f"X_testset shape: {X_testset.shape}")
print(f"Y_testset shape: {Y_testset.shape}")

# =============================================================================
# 1. GAUSSIAN NAIVE BAYES (Basic)
# =============================================================================
print("\n" + "="*60)
print("GAUSSIAN NAIVE BAYES (Basic Model)")
print("="*60)

nb_model = GaussianNB()
nb_model.fit(X_trainset, Y_trainset)
predNB1 = nb_model.predict(X_testset)

# Evaluation
cm = metrics.confusion_matrix(Y_testset, predNB1)
accuracy = metrics.accuracy_score(Y_testset, predNB1)

print(f"\nAccuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(metrics.classification_report(Y_testset, predNB1))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Gaussian Naive Bayes (Basic)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/01_confusion_matrix_gaussian_nb_basic.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 01_confusion_matrix_gaussian_nb_basic.png")
plt.show()

# =============================================================================
# 2. GAUSSIAN NAIVE BAYES (with GridSearchCV)
# =============================================================================
print("\n" + "="*60)
print("GAUSSIAN NAIVE BAYES (with GridSearchCV)")
print("="*60)

from sklearn.model_selection import GridSearchCV

gnb = GaussianNB()
param_gnb = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}

grid_gnb = GridSearchCV(
    estimator=gnb,
    param_grid=param_gnb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_gnb.fit(X_trainset, Y_trainset)

# Show CV results
results = grid_gnb.cv_results_
for i in range(len(results["params"])):
    smooth = results["params"][i]["var_smoothing"]
    cv_acc = results["mean_test_score"][i]
    print(f"var_smoothing={smooth}: CV Accuracy = {cv_acc:.4f}")

print(f"\nBest params: {grid_gnb.best_params_}")
print(f"Best CV Accuracy: {grid_gnb.best_score_:.4f}")

# Predict on test set
y_pred_gnb = grid_gnb.predict(X_testset)

# Evaluation
cm_gnb = metrics.confusion_matrix(Y_testset, y_pred_gnb)
print("\nConfusion Matrix:")
print(cm_gnb)
print("\nClassification Report:")
print(metrics.classification_report(Y_testset, y_pred_gnb))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Gaussian Naive Bayes (GridSearchCV)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/02_confusion_matrix_gaussian_nb_tuned.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 02_confusion_matrix_gaussian_nb_tuned.png")
plt.show()

# =============================================================================
# 3. MULTINOMIAL NAIVE BAYES
# =============================================================================
print("\n" + "="*60)
print("MULTINOMIAL NAIVE BAYES")
print("="*60)

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

# Scale features (MultinomialNB requires non-negative values)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_trainset)
X_test_scaled = scaler.transform(X_testset)

# Try different alpha values
alpha_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

for alpha in alpha_values:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train_scaled, Y_trainset)
    y_pred = mnb.predict(X_test_scaled)
    acc = metrics.accuracy_score(Y_testset, y_pred)
    print(f"alpha={alpha}: Accuracy = {acc:.4f}")

# Final model with last alpha
mnb_final = MultinomialNB(alpha=alpha_values[-1])
mnb_final.fit(X_train_scaled, Y_trainset)
y_pred_mnb = mnb_final.predict(X_test_scaled)

# Evaluation
cm_mnb = metrics.confusion_matrix(Y_testset, y_pred_mnb)
print("\nConfusion Matrix:")
print(cm_mnb)
print("\nClassification Report:")
print(metrics.classification_report(Y_testset, y_pred_mnb, zero_division=0))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Purples',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Multinomial Naive Bayes", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/03_confusion_matrix_multinomial_nb.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 03_confusion_matrix_multinomial_nb.png")
plt.show()

# =============================================================================
# 4. BERNOULLI NAIVE BAYES
# =============================================================================
print("\n" + "="*60)
print("BERNOULLI NAIVE BAYES")
print("="*60)

from sklearn.naive_bayes import BernoulliNB

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_trainset)
X_test_scaled = scaler.transform(X_testset)

# Try different alpha values
alpha_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

for alpha in alpha_values:
    bnb = BernoulliNB(alpha=alpha)
    bnb.fit(X_train_scaled, Y_trainset)
    y_pred = bnb.predict(X_test_scaled)
    acc = metrics.accuracy_score(Y_testset, y_pred)
    print(f"alpha={alpha}: Accuracy = {acc:.4f}")

# Final model with last alpha
bnb_final = BernoulliNB(alpha=alpha_values[-1])
bnb_final.fit(X_train_scaled, Y_trainset)
y_pred_bnb = bnb_final.predict(X_test_scaled)

# Evaluation
cm_bnb = metrics.confusion_matrix(Y_testset, y_pred_bnb)
print("\nConfusion Matrix:")
print(cm_bnb)
print("\nClassification Report:")
print(metrics.classification_report(Y_testset, y_pred_bnb, zero_division=0))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bnb, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Bernoulli Naive Bayes", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/04_confusion_matrix_bernoulli_nb.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 04_confusion_matrix_bernoulli_nb.png")
plt.show()

# =============================================================================
# 5. LOGISTIC REGRESSION
# =============================================================================
print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_trainset)

# Stratified 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
y_pred_lr = cross_val_predict(model_lr, X_scaled, Y_trainset, cv=kf)

# Accuracy
acc_lr = metrics.accuracy_score(Y_trainset, y_pred_lr)
print(f"CV Accuracy = {acc_lr:.4f}")

# Confusion matrix
cm_lr = metrics.confusion_matrix(Y_trainset, y_pred_lr)
print("\nConfusion Matrix:")
print(cm_lr)

# Classification report
print("\nClassification Report:")
print(metrics.classification_report(Y_trainset, y_pred_lr, zero_division=0))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/05_confusion_matrix_logistic_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 05_confusion_matrix_logistic_regression.png")
plt.show()

# =============================================================================
# 6. DECISION TREE
# =============================================================================
print("\n" + "="*60)
print("DECISION TREE")
print("="*60)

from sklearn.tree import DecisionTreeClassifier

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_trainset)

# Stratified 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
y_pred_dt = cross_val_predict(dt_model, X_scaled, Y_trainset, cv=kf)

# Accuracy
acc_dt = metrics.accuracy_score(Y_trainset, y_pred_dt)
print(f"CV Accuracy = {acc_dt:.4f}")

# Confusion matrix
cm_dt = metrics.confusion_matrix(Y_trainset, y_pred_dt)
print("\nConfusion Matrix:")
print(cm_dt)

# Classification report
print("\nClassification Report:")
print(metrics.classification_report(Y_trainset, y_pred_dt, zero_division=0))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Decision Tree", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/06_confusion_matrix_decision_tree.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 06_confusion_matrix_decision_tree.png")
plt.show()

# GridSearchCV for optimal max_depth
print("\nFinding optimal max_depth...")
param_grid = {'max_depth': range(1, 11)}
grid_dt = GridSearchCV(DecisionTreeClassifier(criterion='entropy'),
                       param_grid, cv=5)
grid_dt.fit(X_trainset, Y_trainset)
print(f"Best max_depth: {grid_dt.best_params_['max_depth']}")
print(f"Best CV Score: {grid_dt.best_score_:.4f}")

# =============================================================================
# 7. RANDOM FOREST
# =============================================================================
print("\n" + "="*60)
print("RANDOM FOREST")
print("="*60)

from sklearn.ensemble import RandomForestClassifier

# Model
Model_RF = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=3)
Model_RF.fit(X_trainset, Y_trainset)

# Predictions
predRF = Model_RF.predict(X_testset)

# Accuracy
accuracy_RF = metrics.accuracy_score(Y_testset, predRF)
print(f"Accuracy: {accuracy_RF:.4f}")

# Confusion matrix
cm_RF = metrics.confusion_matrix(Y_testset, predRF)
print("\nConfusion Matrix:")
print(cm_RF)

# Classification report
print("\nClassification Report:")
print(metrics.classification_report(Y_testset, predRF))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_RF, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Random Forest", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/07_confusion_matrix_random_forest.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 07_confusion_matrix_random_forest.png")
plt.show()

# GridSearchCV for optimal max_depth
print("\nFinding optimal max_depth...")
param_grid = {'max_depth': range(1, 11)}
grid_rf = GridSearchCV(DecisionTreeClassifier(criterion='entropy'),
                       param_grid, cv=5)
grid_rf.fit(X_trainset, Y_trainset)
print(f"Best max_depth: {grid_rf.best_params_['max_depth']}")
print(f"Best CV Score: {grid_rf.best_score_:.4f}")

# =============================================================================
# 8. K-NEAREST NEIGHBORS (KNN)
# =============================================================================
print("\n" + "="*60)
print("K-NEAREST NEIGHBORS (KNN)")
print("="*60)

from sklearn.neighbors import KNeighborsClassifier

# Model
Model_KNN = KNeighborsClassifier(n_neighbors=18)
Model_KNN.fit(X_trainset, Y_trainset)

# Predictions
predKNN = Model_KNN.predict(X_testset)

# Accuracy
accuracy_KNN = metrics.accuracy_score(Y_testset, predKNN)
print(f"Accuracy: {accuracy_KNN:.4f}")

# Confusion matrix
cm_KNN = metrics.confusion_matrix(Y_testset, predKNN)
print("\nConfusion Matrix:")
print(cm_KNN)

# Classification report
print("\nClassification Report:")
print(metrics.classification_report(Y_testset, predKNN))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_KNN, annot=True, fmt='d', cmap='Purples',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - K-Nearest Neighbors", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/08_confusion_matrix_knn.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 08_confusion_matrix_knn.png")
plt.show()

# GridSearchCV for optimal n_neighbors
print("\nFinding optimal n_neighbors...")
param_grid = {'n_neighbors': range(1, 21)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_knn.fit(X_trainset, Y_trainset)
print(f"Best n_neighbors: {grid_knn.best_params_['n_neighbors']}")
print(f"Best CV Score: {grid_knn.best_score_:.4f}")

# =============================================================================
# 9. SUPPORT VECTOR MACHINE (SVM)
# =============================================================================
print("\n" + "="*60)
print("SUPPORT VECTOR MACHINE (SVM)")
print("="*60)

from sklearn.svm import SVC

# Model
Model_SVM = SVC(kernel='rbf', gamma='auto', random_state=3)
Model_SVM.fit(X_trainset, Y_trainset)

# Predictions
predSVM = Model_SVM.predict(X_testset)

# Accuracy
accuracy_SVM = metrics.accuracy_score(Y_testset, predSVM)
print(f"Accuracy: {accuracy_SVM:.4f}")

# Confusion matrix
cm_SVM = metrics.confusion_matrix(Y_testset, predSVM)
print("\nConfusion Matrix:")
print(cm_SVM)

# Classification report
print("\nClassification Report:")
print(metrics.classification_report(Y_testset, predSVM))

# Visualization and Save
plt.figure(figsize=(8, 6))
sns.heatmap(cm_SVM, annot=True, fmt='d', cmap='Reds',
            xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Support Vector Machine", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/09_confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 09_confusion_matrix_svm.png")
plt.show()

# =============================================================================
# SUMMARY: ALL MODELS COMPARISON
# =============================================================================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

# Store all results
results_summary = {
    'Model': [
        'Gaussian NB (Basic)',
        'Gaussian NB (Tuned)',
        'Multinomial NB',
        'Bernoulli NB',
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'KNN',
        'SVM'
    ],
    'Accuracy': [
        accuracy,
        metrics.accuracy_score(Y_testset, y_pred_gnb),
        metrics.accuracy_score(Y_testset, y_pred_mnb),
        metrics.accuracy_score(Y_testset, y_pred_bnb),
        acc_lr,
        acc_dt,
        accuracy_RF,
        accuracy_KNN,
        accuracy_SVM
    ]
}

results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

# Create comparison visualization
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors, edgecolor='black')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontweight='bold')

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison - All Algorithms', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('../images/10_model_comparison_summary.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 10_model_comparison_summary.png")
plt.show()

print("\n" + "="*60)
print("ALL VISUALIZATIONS SAVED SUCCESSFULLY!")
print("="*60)
print("\nImages saved in '../images/' directory:")
print("  01_confusion_matrix_gaussian_nb_basic.png")
print("  02_confusion_matrix_gaussian_nb_tuned.png")
print("  03_confusion_matrix_multinomial_nb.png")
print("  04_confusion_matrix_bernoulli_nb.png")
print("  05_confusion_matrix_logistic_regression.png")
print("  06_confusion_matrix_decision_tree.png")
print("  07_confusion_matrix_random_forest.png")
print("  08_confusion_matrix_knn.png")
print("  09_confusion_matrix_svm.png")
print("  10_model_comparison_summary.png")
print("\n" + "="*60)