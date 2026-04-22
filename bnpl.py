
"""
Project: Predicting Credit Default Risk Using POS Loan Patterns
CIS4930 Pattern Recognition
Alexander Nino

This script performs the complete analysis from data loading through 
visualization generation. Run this file after downloading the Home Credit data.

Required files in same directory:
  - application_train.csv
  - POS_CASH_balance.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from matplotlib.patches import Patch

import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

#Create output directories
os.makedirs('figures', exist_ok=True)

#Load the csv files
try:
    #Load main application data
    app_train = pd.read_csv('application_train.csv')
    
    #Load POS loan history
    pos_balance = pd.read_csv('POS_CASH_balance.csv')
    
except FileNotFoundError as e:
    print(f"\nError: Could not find data files")
    print("\nRequired files:")
    print("  - application_train.csv")
    print("  - POS_CASH_balance.csv")
    raise

#Data exploration

#Application overview
print(f"Total loan applications: {len(app_train):,}")
print(f"Total features: {app_train.shape[1]}")
print(f"\nTarget variable distribution:")
print(app_train['TARGET'].value_counts())
default_rate = app_train['TARGET'].mean() * 100
print(f"\nDefault rate: {default_rate:.2f}%")

#Loan data overview
unique_pos_customers = pos_balance['SK_ID_CURR'].nunique()
print(f"Unique customers with POS history: {unique_pos_customers:,}")
pos_coverage = (unique_pos_customers / len(app_train)) * 100
print(f"POS coverage: {pos_coverage:.1f}% of all applicants")

#Check data types and missing values
print(f"\nMissing values in TARGET: {app_train['TARGET'].isna().sum()}")
print(f"Missing values in POS data: {pos_balance.isna().sum().sum():,}")

#POS Feature engineering

#Category 1: POS Loan Frequency
pos_freq = pos_balance.groupby('SK_ID_CURR').agg({
    'SK_ID_PREV': 'nunique',      # Number of unique POS loans
    'MONTHS_BALANCE': 'count'      # Total months of POS history
}).rename(columns={
    'SK_ID_PREV': 'POS_LOAN_COUNT',
    'MONTHS_BALANCE': 'POS_TOTAL_MONTHS'
})

# Calculate loans per year
pos_freq['POS_LOANS_PER_YEAR'] = pos_freq['POS_LOAN_COUNT'] / (pos_freq['POS_TOTAL_MONTHS'] / 12 + 0.001)  #Avoid division by zero

#Category 2: Payment Quality features

#Days Past Due
pos_payment = pos_balance.groupby('SK_ID_CURR').agg({
    'SK_DPD': ['mean', 'max', 'sum'],
    'SK_DPD_DEF': ['mean', 'max', 'sum']
})
pos_payment.columns = ['_'.join(col).upper() for col in pos_payment.columns]

#Calculate on-time payment percentage
pos_balance['IS_LATE'] = (pos_balance['SK_DPD'] > 0).astype(int)
pos_ontime = pos_balance.groupby('SK_ID_CURR')['IS_LATE'].agg(['sum', 'count'])
pos_ontime['POS_ONTIME_PCT'] = (1 - pos_ontime['sum'] / pos_ontime['count']) * 100


#Category 3: Loan Recency

pos_recency = pos_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE'].agg(['max', 'min'])
pos_recency.columns = ['POS_MONTHS_SINCE_LAST', 'POS_MONTHS_SINCE_FIRST']
pos_recency['POS_MONTHS_SINCE_LAST'] = abs(pos_recency['POS_MONTHS_SINCE_LAST'])
pos_recency['POS_MONTHS_SINCE_FIRST'] = abs(pos_recency['POS_MONTHS_SINCE_FIRST'])

#Combine feature categories
pos_features = pos_freq.copy()
pos_features = pos_features.join(pos_payment, how='left')
pos_features = pos_features.join(pos_ontime['POS_ONTIME_PCT'], how='left')
pos_features = pos_features.join(pos_recency, how='left')

#Fill in missing values
pos_features = pos_features.fillna(0)

# Display feature names
print("\nPOS Features:")
print("""
1. Frequency
    POS_LOAN_COUNT
    POS_TOTAL_MONTHS
    POS_LOANS_PER_YEAR
2. Payment Quality
    SK_DPD_MEAN, SK_DPD_MAX, SK_DPD_SUM
    SK_DPD_DEF_MEAN, SK_DPD_DEF_MAX, SK_DPD_DEF_SUM
    POS_ONTIME_PCT
3. Loan Recency
    POS_MONTHS_SINCE_LAST
    POS_MONTHS_SINCE_FIRST

4. BNPL Preference
    BNPL_PROPENSITY_SCORE
""")


#Merge and create final dataset

# Merge POS features
df = app_train.merge(pos_features, left_on='SK_ID_CURR', right_index=True, how='left')

#Create binary flag for having POS history
df['HAS_POS_HISTORY'] = df['POS_LOAN_COUNT'].notna().astype(int)

#Fill missing POS features (customers without POS history)
pos_cols = [col for col in df.columns if col.startswith('POS_') or col.startswith('SK_DPD')]
df[pos_cols] = df[pos_cols].fillna(0)

print(f"Merged dataset shape: {df.shape}")
print(f"Customers with POS history: {df['HAS_POS_HISTORY'].sum():,} ({df['HAS_POS_HISTORY'].mean()*100:.1f}%)")


baseline_features = [
    'AMT_INCOME_TOTAL',       # Income
    'AMT_CREDIT',             # Loan amount
    'AMT_ANNUITY',            # Annuity
    'DAYS_BIRTH',             # Age (negative days)
    'DAYS_EMPLOYED',          # Employment length (negative days)
    'CNT_CHILDREN',           # Number of children
    'CNT_FAM_MEMBERS',        # Family size
]

print(f"Baseline features: {len(baseline_features)}")

#Propensity score
#Normalize key features
scaler_bnpl = MinMaxScaler()
bnpl_components = ['POS_LOAN_COUNT', 'POS_LOANS_PER_YEAR', 'SK_DPD_MEAN', 'POS_MONTHS_SINCE_LAST']

df_bnpl_scaled = pd.DataFrame(
    scaler_bnpl.fit_transform(df[bnpl_components]),
    columns=bnpl_components,
    index=df.index
)

#Create weighted BNPL score
df['BNPL_PROPENSITY_SCORE'] = (
    df_bnpl_scaled['POS_LOAN_COUNT'] * 0.3 +           # Frequency weight
    df_bnpl_scaled['POS_LOANS_PER_YEAR'] * 0.2 +      # Intensity weight
    df_bnpl_scaled['SK_DPD_MEAN'] * 0.3 +             # Payment quality weight
    (1 - df_bnpl_scaled['POS_MONTHS_SINCE_LAST']) * 0.2  # Recency weight (inverted)
)
print(f"Score range: {df['BNPL_PROPENSITY_SCORE'].min():.3f} to {df['BNPL_PROPENSITY_SCORE'].max():.3f}")


df = df.replace([np.inf, -np.inf], np.nan)

#Fill baseline features with median
for col in baseline_features:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

#Replace extreme DAYS values
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())

#Exploratory Data Analysis

print("\n--- DEFAULT RATE BY POS LOAN COUNT ---")
pos_default = df.groupby('POS_LOAN_COUNT')['TARGET'].agg(['mean', 'count'])
pos_default = pos_default[pos_default['count'] >= 100]  #Only groups with 100+ samples
pos_default.columns = ['Default_Rate', 'Sample_Size']
pos_default['Default_Rate'] = pos_default['Default_Rate'] * 100

print(pos_default.head(10).to_string())

print("\n--- DEFAULT RATE BY POS HISTORY ---")
has_pos_stats = df.groupby('HAS_POS_HISTORY')['TARGET'].agg(['mean', 'count'])
has_pos_stats.columns = ['Default_Rate', 'Sample_Size']
has_pos_stats['Default_Rate'] = has_pos_stats['Default_Rate'] * 100
print(has_pos_stats.to_string())

print("\n--- TOP FEATURE CORRELATIONS WITH DEFAULT ---")
all_features = baseline_features + pos_cols + ['BNPL_PROPENSITY_SCORE']
correlations = df[all_features].corrwith(df['TARGET']).sort_values(ascending=False)
print("\nTop 10 Positive Correlations:")
print(correlations.head(10).to_string())

#Train/Test

#Define feature sets
baseline_feature_cols = [col for col in baseline_features if col in df.columns]
enhanced_feature_cols = baseline_feature_cols + pos_cols + ['BNPL_PROPENSITY_SCORE']

#Remove duplicates
enhanced_feature_cols = list(dict.fromkeys(enhanced_feature_cols))

# Prepare datasets
X_baseline = df[baseline_feature_cols].copy()
X_enhanced = df[enhanced_feature_cols].copy()
y = df['TARGET'].copy()

#Split data 80/20

X_baseline_train, X_baseline_test, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42, stratify=y
)

X_enhanced_train, X_enhanced_test, _, _ = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_baseline_train):,} samples")
print(f"Test set: {len(X_baseline_test):,} samples")
print(f"Default rate in train: {y_train.mean()*100:.2f}%")
print(f"Default rate in test: {y_test.mean()*100:.2f}%")

#Scale features
scaler_baseline = StandardScaler()
scaler_enhanced = StandardScaler()

X_baseline_train_scaled = scaler_baseline.fit_transform(X_baseline_train)
X_baseline_test_scaled = scaler_baseline.transform(X_baseline_test)

X_enhanced_train_scaled = scaler_enhanced.fit_transform(X_enhanced_train)
X_enhanced_test_scaled = scaler_enhanced.transform(X_enhanced_test)

#Build and evaluate baseline
results = {}

#logistic regression
lr_baseline = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_baseline.fit(X_baseline_train_scaled, y_train)

y_pred_lr_base = lr_baseline.predict(X_baseline_test_scaled)
y_proba_lr_base = lr_baseline.predict_proba(X_baseline_test_scaled)[:, 1]

results['Logistic_Baseline'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr_base),
    'precision': precision_score(y_test, y_pred_lr_base),
    'recall': recall_score(y_test, y_pred_lr_base),
    'f1': f1_score(y_test, y_pred_lr_base),
    'auc': roc_auc_score(y_test, y_proba_lr_base)
}

#Decision Tree
dt_baseline = DecisionTreeClassifier(
    random_state=42, max_depth=10, min_samples_split=100, class_weight='balanced'
)
dt_baseline.fit(X_baseline_train_scaled, y_train)

y_pred_dt_base = dt_baseline.predict(X_baseline_test_scaled)
y_proba_dt_base = dt_baseline.predict_proba(X_baseline_test_scaled)[:, 1]

results['DecisionTree_Baseline'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt_base),
    'precision': precision_score(y_test, y_pred_dt_base),
    'recall': recall_score(y_test, y_pred_dt_base),
    'f1': f1_score(y_test, y_pred_dt_base),
    'auc': roc_auc_score(y_test, y_proba_dt_base)
}


#Random Forest
rf_baseline = RandomForestClassifier(
    n_estimators=100, random_state=42, max_depth=10,
    min_samples_split=100, class_weight='balanced', n_jobs=-1
)
rf_baseline.fit(X_baseline_train_scaled, y_train)

y_pred_rf_base = rf_baseline.predict(X_baseline_test_scaled)
y_proba_rf_base = rf_baseline.predict_proba(X_baseline_test_scaled)[:, 1]

results['RandomForest_Baseline'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf_base),
    'precision': precision_score(y_test, y_pred_rf_base),
    'recall': recall_score(y_test, y_pred_rf_base),
    'f1': f1_score(y_test, y_pred_rf_base),
    'auc': roc_auc_score(y_test, y_proba_rf_base)
}

#Build and evaluate enhanced feature models

#Logistic Regression
lr_enhanced = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_enhanced.fit(X_enhanced_train_scaled, y_train)

y_pred_lr_enh = lr_enhanced.predict(X_enhanced_test_scaled)
y_proba_lr_enh = lr_enhanced.predict_proba(X_enhanced_test_scaled)[:, 1]

results['Logistic_Enhanced'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr_enh),
    'precision': precision_score(y_test, y_pred_lr_enh),
    'recall': recall_score(y_test, y_pred_lr_enh),
    'f1': f1_score(y_test, y_pred_lr_enh),
    'auc': roc_auc_score(y_test, y_proba_lr_enh)
}

#Decision Tree
dt_enhanced = DecisionTreeClassifier(
    random_state=42, max_depth=10, min_samples_split=100, class_weight='balanced'
)
dt_enhanced.fit(X_enhanced_train_scaled, y_train)

y_pred_dt_enh = dt_enhanced.predict(X_enhanced_test_scaled)
y_proba_dt_enh = dt_enhanced.predict_proba(X_enhanced_test_scaled)[:, 1]

results['DecisionTree_Enhanced'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt_enh),
    'precision': precision_score(y_test, y_pred_dt_enh),
    'recall': recall_score(y_test, y_pred_dt_enh),
    'f1': f1_score(y_test, y_pred_dt_enh),
    'auc': roc_auc_score(y_test, y_proba_dt_enh)
}

#Random Forest-
rf_enhanced = RandomForestClassifier(
    n_estimators=100, random_state=42, max_depth=10,
    min_samples_split=100, class_weight='balanced', n_jobs=-1
)
rf_enhanced.fit(X_enhanced_train_scaled, y_train)

y_pred_rf_enh = rf_enhanced.predict(X_enhanced_test_scaled)
y_proba_rf_enh = rf_enhanced.predict_proba(X_enhanced_test_scaled)[:, 1]

results['RandomForest_Enhanced'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf_enh),
    'precision': precision_score(y_test, y_pred_rf_enh),
    'recall': recall_score(y_test, y_pred_rf_enh),
    'f1': f1_score(y_test, y_pred_rf_enh),
    'auc': roc_auc_score(y_test, y_proba_rf_enh)
}

#Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': enhanced_feature_cols,
    'importance': rf_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

#Summary Statistics
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print("\nModel Performance")
print(results_df.to_string())

print("\nPercentage improvement")
for model in ['Logistic', 'DecisionTree', 'RandomForest']:
    baseline_auc = results[f'{model}_Baseline']['auc']
    enhanced_auc = results[f'{model}_Enhanced']['auc']
    improvement = enhanced_auc - baseline_auc
    pct_improvement = (improvement / baseline_auc) * 100
    print(f"  {model:15s}: ({pct_improvement:+.2f}%)")

#Visualizationn
colors_viz = {
    'primary': '#028090',
    'secondary': '#02C39A',
    'accent': '#e74c3c',
    'baseline': '#3498db',
    'enhanced': '#e74c3c'
}

#Figure 1: Model Performance comparison

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#Subplot 1: AUC Comparison
models = ['Logistic', 'DecisionTree', 'RandomForest']
baseline_aucs = [results[f'{m}_Baseline']['auc'] for m in models]
enhanced_aucs = [results[f'{m}_Enhanced']['auc'] for m in models]

x = np.arange(len(models))
width = 0.35

bars1 = axes[0].bar(x - width/2, baseline_aucs, width, label='Baseline (No POS)',
                    color=colors_viz['baseline'], alpha=0.8)
bars2 = axes[0].bar(x + width/2, enhanced_aucs, width, label='Enhanced (With POS)',
                    color=colors_viz['enhanced'], alpha=0.8)

axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[0].set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
axes[0].set_title('Model Performance: Baseline vs Enhanced', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0.5, max(enhanced_aucs) * 1.1])

#Add value labels
for i, (b, e) in enumerate(zip(baseline_aucs, enhanced_aucs)):
    axes[0].text(i - width/2, b + 0.01, f'{b:.3f}', ha='center', fontsize=10)
    axes[0].text(i + width/2, e + 0.01, f'{e:.3f}', ha='center', fontsize=10)

#Subplot 2: Improvement Percentage
improvements = [(e - b) / b * 100 for b, e in zip(baseline_aucs, enhanced_aucs)]

bars = axes[1].bar(models, improvements,
                   color=[colors_viz['secondary'] if imp > 0 else '#e67e22' for imp in improvements],
                   alpha=0.8)
axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUC Improvement (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Relative Improvement from POS Features', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)

#Add value labels
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    axes[1].text(i, imp + 0.5, f'{imp:+.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


#Figure 2: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))

top_features = feature_importance.head(10)
colors_feat = [colors_viz['enhanced'] if ('POS' in feat or 'BNPL' in feat or 'DPD' in feat)
               else colors_viz['baseline'] for feat in top_features['feature']]

bars = ax.barh(top_features['feature'], top_features['importance'], color=colors_feat, alpha=0.8)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Feature Importance (Random Forest Enhanced)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

#Add value labels
for i, (bar, imp) in enumerate(zip(bars, top_features['importance'])):
    ax.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=9)


legend_elements = [
    Patch(facecolor=colors_viz['enhanced'], alpha=0.8, label='POS/BNPL Features'),
    Patch(facecolor=colors_viz['baseline'], alpha=0.8, label='Traditional Features')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

#Figure 3: Default rate by POS loan count
pos_default_viz = df.groupby('POS_LOAN_COUNT')['TARGET'].agg(['mean', 'count'])
pos_default_viz = pos_default_viz[pos_default_viz['count'] >= 100]  # Filter for significance
pos_default_viz = pos_default_viz.head(8)  # Limit to first 8 for clarity
pos_default_viz['mean'] = pos_default_viz['mean'] * 100  # Convert to percentage

fig, ax1 = plt.subplots(figsize=(10, 6))

#Plot default rate
color = colors_viz['enhanced']
ax1.plot(pos_default_viz.index, pos_default_viz['mean'], marker='o', linewidth=2.5,
         markersize=8, color=color, label='Default Rate')
ax1.set_xlabel('Number of POS Loans', fontsize=12, fontweight='bold')
ax1.set_ylabel('Default Rate (%)', fontsize=12, fontweight='bold', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(alpha=0.3)

#Plot sample size on second y-axis
ax2 = ax1.twinx()
color = colors_viz['baseline']
ax2.bar(pos_default_viz.index, pos_default_viz['count'], alpha=0.3, color=color, label='Sample Size')
ax2.set_ylabel('Number of Customers', fontsize=12, fontweight='bold', color=color)
ax2.tick_params(axis='y', labelcolor=color)

#Title and legend
ax1.set_title('Default Rate Increases with POS Loan Frequency', fontsize=14, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))

plt.tight_layout()
plt.savefig('figures/default_by_pos_count.png', dpi=300, bbox_inches='tight')
plt.close()


# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_rf_enh)
tn, fp, fn, tp = cm.ravel()

# Create figure with more space at bottom
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            square=True, linewidths=2, linecolor='white',
            annot_kws={'size': 16, 'weight': 'bold'})

# Labels
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Actual Label', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Confusion Matrix - Random Forest Enhanced\n(Best Model)', 
             fontsize=15, fontweight='bold', pad=20)

# Set tick labels
ax.set_xticklabels(['No Default (0)', 'Default (1)'], fontsize=12)
ax.set_yticklabels(['No Default (0)', 'Default (1)'], fontsize=12, rotation=0)

# Move up the plot to make room for text
plt.subplots_adjust(bottom=0.25)

# Add performance metrics as text below the matrix
metrics_text = f"""Classification Results (Test Set: {len(y_test):,} samples):

True Negatives (TN):   {tn:,}      Correctly predicted "No Default"
False Positives (FP):  {fp:,}      Predicted "Default" but actually paid back
False Negatives (FN):  {fn:,}      Predicted "No Default" but actually defaulted
True Positives (TP):   {tp:,}      Correctly predicted "Default"

Performance Metrics:
  Accuracy:   {accuracy_score(y_test, y_pred_rf_enh)*100:.1f}%   
  Precision:  {precision_score(y_test, y_pred_rf_enh)*100:.1f}%   
  Recall:     {recall_score(y_test, y_pred_rf_enh)*100:.1f}%   
  AUC:        {roc_auc_score(y_test, y_proba_rf_enh):.3f}   """


plt.figtext(0.5, 0.08, metrics_text, ha='center', fontsize=9.5, 
            family='monospace', 
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', 
                     edgecolor='#333333', linewidth=1.5, alpha=0.9))

plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# After training rf_enhanced
print("\nTop 10 Feature Importances:")
print("="*50)

feature_importance = pd.DataFrame({
    'Feature': enhanced_feature_cols,
    'Importance': rf_enhanced.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']:30s} {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")
