# ALY6020 - Predictive Analytics | Week 3 Assignment
# Loan Default Prediction - SecureBank
# Classification Models: Logistic Regression & SVM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score, classification_report)
from sklearn.feature_selection import RFE

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('loan_default.csv')
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df['Loan_Default'].value_counts())
print(f"\nDefault rate: {df['Loan_Default'].value_counts(normalize=True)['Yes']:.1%}")

# ============================================================
# 2. DATA CLEANING
# ============================================================

# --- Loan_Duration_Months (76 missing, numeric) ---
# Loan duration is positively skewed and relates to loan size/purpose.
# Median is more robust than mean to outliers in financial data.
# Using median to avoid inflating duration for short-term loans.
duration_median = df['Loan_Duration_Months'].median()
df.loc[df['Loan_Duration_Months'].isnull(), 'Loan_Duration_Months'] = duration_median
print(f"\nLoan_Duration_Months: filled 76 missing values with median ({duration_median})")

# --- Loan_Amount (127 missing, numeric; note: negative values present) ---
# Min value is -5410 DM - likely a data entry error or credit/refund scenario.
# Negative amounts are not valid loan amounts; we cap them at 0 before imputing.
# For missing values, median is preferred over mean because the distribution is
# right-skewed (max=19,972 DM) and mean would overestimate typical loan size.
df['Loan_Amount'] = df['Loan_Amount'].clip(lower=0)
amount_median = df['Loan_Amount'].median()
df.loc[df['Loan_Amount'].isnull(), 'Loan_Amount'] = amount_median
print(f"Loan_Amount: negative values clipped to 0; 127 missing filled with median ({amount_median:.0f} DM)")

# --- Age (103 missing, numeric) ---
# Age is relatively normally distributed (mean=35, median=33).
# However, age in lending decisions can have non-linear effects.
# Median imputation preserves the central tendency without skewing
# the distribution toward older borrowers in the right tail.
age_median = df['Age'].median()
df.loc[df['Age'].isnull(), 'Age'] = age_median
print(f"Age: filled 103 missing values with median ({age_median})")

# --- Employment_Duration (102 missing, categorical) ---
# Employment duration is an ordinal categorical variable (unemployed < 1yr < 4yr < 7yr+).
# Missing employment info likely indicates unemployed or unstated status.
# We impute with the mode ('1 - 4 years') since it is the most common category
# and avoids assuming worst-case (unemployed) or best-case (>= 7 years).
emp_mode = df['Employment_Duration'].mode()[0]
df.loc[df['Employment_Duration'].isnull(), 'Employment_Duration'] = emp_mode
print(f"Employment_Duration: filled 102 missing values with mode ('{emp_mode}')")

# Verify no remaining missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("No missing values remain." if df.isnull().sum().sum() == 0 else "WARNING: missing values still present")

# ============================================================
# 3. ENCODING
# ============================================================

# Target variable: encode Yes=1, No=0
df['Loan_Default'] = (df['Loan_Default'] == 'Yes').astype(int)

# Ordinal variables - preserve natural order with manual mapping
# These have a meaningful progression that label encoding captures correctly.
checking_map = {'< 0 DM': 0, '0 - 200 DM': 1, '>= 200 DM': 2, 'no checking account': 3}
savings_map = {'< 100 DM': 0, '100 - 500 DM': 1, '500 - 1000 DM': 2,
               '>= 1000 DM': 3, 'unknown / no savings': 4}
employment_map = {'unemployed': 0, '< 1 year': 1, '1 - 4 years': 2,
                  '4 - 7 years': 3, '>= 7 years': 4}

df['Checking_Account_Status'] = df['Checking_Account_Status'].map(checking_map)
df['Savings_Account_Status'] = df['Savings_Account_Status'].map(savings_map)
df['Employment_Duration'] = df['Employment_Duration'].map(employment_map)

# Binary variables - simple label encoding (2 unique values)
binary_cols = ['Has_Telephone', 'Is_Foreign_Worker']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Nominal variables - one-hot encoding (no inherent order)
nominal_cols = ['Credit_History', 'Loan_Purpose', 'Personal_Status',
                'Other_Debtors_Guarantors', 'Property_Type',
                'Other_Installment_Plans', 'Housing_Type', 'Job_Type']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

print("\nDataset shape after encoding:", df.shape)

# ============================================================
# 4. EDA VISUALIZATIONS
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')

# --- Plot 1: Default Rate by Checking Account Status ---
fig, ax = plt.subplots(figsize=(8, 5))
checking_labels = {0: '< 0 DM', 1: '0-200 DM', 2: '>= 200 DM', 3: 'No Account'}
df['Checking_Label'] = df['Checking_Account_Status'].map(checking_labels)
default_by_checking = df.groupby('Checking_Label')['Loan_Default'].mean().reset_index()
default_by_checking.columns = ['Checking Account Status', 'Default Rate']
default_by_checking = default_by_checking.sort_values('Default Rate', ascending=False)
bars = ax.bar(default_by_checking['Checking Account Status'],
              default_by_checking['Default Rate'],
              color=['#d73027', '#fc8d59', '#fee090', '#91cf60'])
ax.set_title('Default Rate by Checking Account Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Checking Account Status')
ax.set_ylabel('Default Rate')
ax.set_ylim(0, 0.6)
for bar, val in zip(bars, default_by_checking['Default Rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.1%}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('plot1_default_by_checking.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot 1 saved: Default Rate by Checking Account Status")
df.drop('Checking_Label', axis=1, inplace=True)

# --- Plot 2: Loan Amount Distribution by Default Status ---
fig, ax = plt.subplots(figsize=(8, 5))
default_yes = df[df['Loan_Default'] == 1]['Loan_Amount']
default_no = df[df['Loan_Default'] == 0]['Loan_Amount']
ax.hist(default_no, bins=40, alpha=0.6, label='No Default', color='#2166ac', edgecolor='white')
ax.hist(default_yes, bins=40, alpha=0.6, label='Default', color='#d73027', edgecolor='white')
ax.axvline(default_no.median(), color='#2166ac', linestyle='--', linewidth=1.5,
           label=f'No Default Median: {default_no.median():.0f} DM')
ax.axvline(default_yes.median(), color='#d73027', linestyle='--', linewidth=1.5,
           label=f'Default Median: {default_yes.median():.0f} DM')
ax.set_title('Loan Amount Distribution by Default Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Loan Amount (DM)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('plot2_loan_amount_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 saved: Loan Amount Distribution by Default Status")

# --- Plot 3: Age Distribution by Default Status ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(default_no_age := df[df['Loan_Default'] == 0]['Age'],
        bins=30, alpha=0.6, label='No Default', color='#2166ac', edgecolor='white')
ax.hist(default_yes_age := df[df['Loan_Default'] == 1]['Age'],
        bins=30, alpha=0.6, label='Default', color='#d73027', edgecolor='white')
ax.axvline(default_no_age.median(), color='#2166ac', linestyle='--', linewidth=1.5,
           label=f'No Default Median Age: {default_no_age.median():.0f}')
ax.axvline(default_yes_age.median(), color='#d73027', linestyle='--', linewidth=1.5,
           label=f'Default Median Age: {default_yes_age.median():.0f}')
ax.set_title('Age Distribution by Default Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Age (Years)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('plot3_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 saved: Age Distribution by Default Status")

# ============================================================
# 5. MODEL BUILDING
# ============================================================

# Prepare features and target
X = df.drop('Loan_Default', axis=1)
y = df['Loan_Default']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} records")
print(f"Test set:     {X_test.shape[0]} records")

# Scale features (required for SVM; harmless for logistic regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

def evaluate_model(name, y_true, y_pred, y_prob=None):
    """Print full evaluation metrics for a model."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print('='*50)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC:       {auc:.4f}")
    return cm

# ============================================================
# MODEL 1: Logistic Regression - Baseline (All Features)
# ============================================================
print("\n\n--- MODEL 1: Logistic Regression (Baseline - All Features) ---")
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train_sc, y_train)
y_pred_lr1 = lr_baseline.predict(X_test_sc)
y_prob_lr1 = lr_baseline.predict_proba(X_test_sc)[:, 1]

cm1 = evaluate_model("Model 1: Logistic Regression (Baseline)", y_test, y_pred_lr1, y_prob_lr1)

# Coefficient interpretation - top 3 significant features
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_baseline.coef_[0],
    'Odds_Ratio': np.exp(lr_baseline.coef_[0])
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Features by Coefficient Magnitude:")
print(coef_df.head(10).to_string(index=False))

print("\nOdds Ratio Interpretation (top 3):")
top3 = coef_df.head(3)
for _, row in top3.iterrows():
    direction = "increases" if row['Odds_Ratio'] > 1 else "decreases"
    print(f"  {row['Feature']}: OR={row['Odds_Ratio']:.3f} — "
          f"each unit increase {direction} default odds by {abs(row['Odds_Ratio']-1)*100:.1f}%")

# ============================================================
# MODEL 2: Optimized Logistic Regression (Stepwise - RFE)
# ============================================================
print("\n\n--- MODEL 2: Optimized Logistic Regression (RFE Feature Selection) ---")

# Use RFE (Recursive Feature Elimination) - equivalent to backward stepwise
# Start with all features and iteratively remove least important
lr_for_rfe = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=lr_for_rfe, n_features_to_select=15, step=1)
rfe.fit(X_train_sc, y_train)

selected_features = X.columns[rfe.support_].tolist()
print(f"Selected {len(selected_features)} features via RFE:")
print(selected_features)

X_train_rfe = X_train_sc[:, rfe.support_]
X_test_rfe = X_test_sc[:, rfe.support_]

lr_optimized = LogisticRegression(max_iter=1000, random_state=42)
lr_optimized.fit(X_train_rfe, y_train)
y_pred_lr2 = lr_optimized.predict(X_test_rfe)
y_prob_lr2 = lr_optimized.predict_proba(X_test_rfe)[:, 1]

cm2 = evaluate_model("Model 2: Optimized Logistic Regression (RFE)", y_test, y_pred_lr2, y_prob_lr2)

# ============================================================
# MODEL 3: SVM (RBF Kernel)
# ============================================================
print("\n\n--- MODEL 3: SVM (RBF Kernel) ---")
# RBF kernel chosen because loan default relationships are likely non-linear
# (e.g., the impact of loan amount varies by checking account status)
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
svm_rbf.fit(X_train_sc, y_train)
y_pred_svm = svm_rbf.predict(X_test_sc)
y_prob_svm = svm_rbf.predict_proba(X_test_sc)[:, 1]

cm3 = evaluate_model("Model 3: SVM (RBF Kernel)", y_test, y_pred_svm, y_prob_svm)

# ============================================================
# ROC CURVE - All Three Models
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for name, y_prob in [("LR Baseline", y_prob_lr1),
                     ("LR Optimized", y_prob_lr2),
                     ("SVM RBF", y_prob_svm)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plot4_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot 4 saved: ROC Curves")

# ============================================================
# MODEL COMPARISON SUMMARY
# ============================================================
print("\n\n{'='*60}")
print("MODEL COMPARISON SUMMARY")
print('='*60)
models = {
    'LR Baseline': (y_pred_lr1, y_prob_lr1),
    'LR Optimized': (y_pred_lr2, y_prob_lr2),
    'SVM RBF': (y_pred_svm, y_prob_svm)
}

summary = []
for name, (y_pred, y_prob) in models.items():
    summary.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_prob), 4)
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

print("\n\nScript complete. Output plots saved to working directory.")
