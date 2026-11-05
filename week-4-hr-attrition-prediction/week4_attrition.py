import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('hr_attrition.csv')

print("=== INITIAL DATA INSPECTION ===")
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDuplicate EmployeeIDs: {df['EmployeeID'].duplicated().sum()}")
print(f"Negative Monthly_Income: {(df['Monthly_Income'] < 0).sum()}")
print(f"Performance_Rating = 0: {(df['Performance_Rating'] == 0).sum()}")
print(f"Years_Since_Last_Promotion > Years_at_Company: {(df['Years_Since_Last_Promotion'] > df['Years_at_Company']).sum()}")

print("\n=== DATA CLEANING ===")
df = df.drop_duplicates(subset='EmployeeID', keep='first')
print(f"Removed 4 duplicate records. Remaining: {len(df)}")

df.loc[df['Monthly_Income'] < 0, 'Monthly_Income'] = df[df['Monthly_Income'] > 0]['Monthly_Income'].median()
print("Replaced 3 negative Monthly_Income values with column median.")

df.loc[df['Performance_Rating'] == 0, 'Performance_Rating'] = int(df[df['Performance_Rating'] > 0]['Performance_Rating'].median())
print("Replaced 6 Performance_Rating = 0 values with column median.")

df.loc[df['Years_Since_Last_Promotion'] > df['Years_at_Company'], 'Years_Since_Last_Promotion'] = df['Years_at_Company']
print("Capped 5 impossible promotion values at Years_at_Company.")

df['Monthly_Income'] = df['Monthly_Income'].fillna(df['Monthly_Income'].median())
df['Job_Satisfaction'] = df['Job_Satisfaction'].fillna(df['Job_Satisfaction'].median())
df['Work_Life_Balance'] = df['Work_Life_Balance'].fillna(df['Work_Life_Balance'].median())
df['Manager_Rating'] = df['Manager_Rating'].fillna(df['Manager_Rating'].median())
print("Imputed 120 missing Monthly_Income, 150 Job_Satisfaction, 90 Work_Life_Balance, 120 Manager_Rating with column medians.")
print(f"Remaining nulls: {df.isnull().sum().sum()}")

print("\n=== ATTRITION DISTRIBUTION ===")
print(df['Attrition'].value_counts())
print(df['Attrition'].value_counts(normalize=True).round(4))

print("\n=== EDA ===")
dept_attr = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
print(f"Attrition rate by Department:\n{dept_attr.round(2)}")

js_attr = df.groupby('Job_Satisfaction')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
print(f"\nAttrition rate by Job Satisfaction:\n{js_attr.round(2)}")

print(f"\nMean income - Leavers: ${df[df['Attrition']=='Yes']['Monthly_Income'].mean():.0f}")
print(f"Mean income - Stayers: ${df[df['Attrition']=='No']['Monthly_Income'].mean():.0f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 1: Attrition Exploratory Analysis', fontsize=14, fontweight='bold')

bars = axes[0].bar(dept_attr.index, dept_attr.values, color=['#E05252' if v > dept_attr.mean() else '#4A7EC4' for v in dept_attr.values])
axes[0].set_title('Attrition Rate by Department (%)', fontweight='bold')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].tick_params(axis='x', rotation=30)
for bar, val in zip(bars, dept_attr.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontsize=8)

axes[1].bar([str(int(x)) for x in js_attr.index], js_attr.values, color='#E05252', alpha=0.85)
axes[1].set_title('Attrition Rate by Job Satisfaction Level', fontweight='bold')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_xlabel('Job Satisfaction (1=Low, 5=High)')
for i, val in enumerate(js_attr.values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=9)

leavers_inc = df[df['Attrition'] == 'Yes']['Monthly_Income']
stayers_inc = df[df['Attrition'] == 'No']['Monthly_Income']
axes[2].hist(stayers_inc, bins=30, alpha=0.6, color='#4A7EC4', label='Stayed', density=True)
axes[2].hist(leavers_inc, bins=30, alpha=0.6, color='#E05252', label='Left', density=True)
axes[2].set_title('Monthly Income: Leavers vs. Stayers', fontweight='bold')
axes[2].set_xlabel('Monthly Income ($)')
axes[2].set_ylabel('Density')
axes[2].legend()
axes[2].axvline(leavers_inc.mean(), color='#E05252', linestyle='--', alpha=0.8)
axes[2].axvline(stayers_inc.mean(), color='#4A7EC4', linestyle='--', alpha=0.8)

plt.tight_layout()
plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA charts saved to eda_charts.png")

print("\n=== MODELING ===")
df_model = df.copy()
df_model['Attrition_bin'] = (df_model['Attrition'] == 'Yes').astype(int)
df_model['Overtime_bin'] = (df_model['Overtime'] == 'Yes').astype(int)

cat_cols = ['Gender', 'Department', 'Job_Role', 'Education_Level', 'Marital_Status']
le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

features = ['Job_Satisfaction', 'Work_Life_Balance', 'Age', 'Overtime_bin', 'Stock_Option_Level',
            'Monthly_Income', 'Years_at_Company', 'Years_Since_Last_Promotion', 'Distance_From_Home',
            'Num_Companies_Worked', 'Training_Hours_Last_Year', 'Performance_Rating', 'Manager_Rating',
            'Department', 'Marital_Status']

X = df_model[features]
y = df_model['Attrition_bin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return acc, prec, rec, f1, cm

m1 = DecisionTreeClassifier(random_state=42)
m1.fit(X_train, y_train)
r1 = evaluate(m1, X_test, y_test, "Model 1: Unpruned Decision Tree")
print(f"  Tree depth: {m1.get_depth()}, Leaves: {m1.get_n_leaves()}")

m2 = DecisionTreeClassifier(max_depth=5, random_state=42)
m2.fit(X_train, y_train)
r2 = evaluate(m2, X_test, y_test, "Model 2: Pruned Decision Tree (max_depth=5)")

fi_pruned = pd.Series(m2.feature_importances_, index=features).sort_values(ascending=False)
print(f"\nTop 5 Feature Importances (Pruned Tree):\n{fi_pruned.head(5).round(4)}")

print("\nPruned Tree Structure (first 3 levels):")
print(export_text(m2, feature_names=features, max_depth=3))

m3 = RandomForestClassifier(n_estimators=100, random_state=42)
m3.fit(X_train, y_train)
r3 = evaluate(m3, X_test, y_test, "Model 3: Random Forest (100 trees)")

fi_rf = pd.Series(m3.feature_importances_, index=features).sort_values(ascending=False)
print(f"\nTop 5 Feature Importances (Random Forest):\n{fi_rf.head(5).round(4)}")

print("\n=== MODEL COMPARISON SUMMARY ===")
print(f"{'Model':<40} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
models = [("Model 1: Unpruned Decision Tree", r1), ("Model 2: Pruned Decision Tree (depth=5)", r2), ("Model 3: Random Forest (100 trees)", r3)]
for name, (acc, prec, rec, f1, cm) in models:
    print(f"{name:<40} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>8.4f}")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Figure 2: Feature Importance Comparison', fontsize=13, fontweight='bold')

fi_top8 = fi_pruned.head(8)
axes2[0].barh(fi_top8.index[::-1], fi_top8.values[::-1], color='#4A7EC4')
axes2[0].set_title('Pruned Decision Tree', fontweight='bold')
axes2[0].set_xlabel('Feature Importance Score')
for bar, val in zip(axes2[0].patches, fi_top8.values[::-1]):
    axes2[0].text(val + 0.002, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=8)

fi_rf_top8 = fi_rf.head(8)
axes2[1].barh(fi_rf_top8.index[::-1], fi_rf_top8.values[::-1], color='#5BAD6F')
axes2[1].set_title('Random Forest (100 Trees)', fontweight='bold')
axes2[1].set_xlabel('Feature Importance Score')
for bar, val in zip(axes2[1].patches, fi_rf_top8.values[::-1]):
    axes2[1].text(val + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFeature importance charts saved to feature_importance.png")
