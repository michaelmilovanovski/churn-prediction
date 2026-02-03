"""Generate visualization images for the churn prediction project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading data...")
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# ============================================================
# 1. CHURN DISTRIBUTION
# ============================================================
print("Creating churn distribution chart...")
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
churn_counts = df['Churn'].value_counts()
bars = ax.bar(['Retained', 'Churned'], [churn_counts['No'], churn_counts['Yes']],
              color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, count in zip(bars, [churn_counts['No'], churn_counts['Yes']]):
    height = bar.get_height()
    pct = count / len(df) * 100
    ax.annotate(f'{count:,}\n({pct:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Number of Customers', fontsize=12)
ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, max(churn_counts) * 1.15)
plt.tight_layout()
plt.savefig('images/churn_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: images/churn_distribution.png")

# ============================================================
# 2. MODEL COMPARISON
# ============================================================
print("Training models for comparison chart...")

# Train models
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                          random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Get predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = {
    'Logistic Regression': {
        'Accuracy': accuracy_score(y_test, lr_model.predict(X_test)),
        'Precision': precision_score(y_test, lr_model.predict(X_test)),
        'Recall': recall_score(y_test, lr_model.predict(X_test)),
        'F1-Score': f1_score(y_test, lr_model.predict(X_test)),
        'ROC-AUC': roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
    },
    'Random Forest': {
        'Accuracy': accuracy_score(y_test, rf_model.predict(X_test)),
        'Precision': precision_score(y_test, rf_model.predict(X_test)),
        'Recall': recall_score(y_test, rf_model.predict(X_test)),
        'F1-Score': f1_score(y_test, rf_model.predict(X_test)),
        'ROC-AUC': roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    },
    'XGBoost': {
        'Accuracy': accuracy_score(y_test, xgb_model.predict(X_test)),
        'Precision': precision_score(y_test, xgb_model.predict(X_test)),
        'Recall': recall_score(y_test, xgb_model.predict(X_test)),
        'F1-Score': f1_score(y_test, xgb_model.predict(X_test)),
        'ROC-AUC': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    }
}

print("Creating model comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

colors = ['#3498db', '#2ecc71', '#e74c3c']
models = ['Logistic Regression', 'Random Forest', 'XGBoost']

for i, (model, color) in enumerate(zip(models, colors)):
    values = [results[model][m] for m in metrics]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width, label=model, color=color, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('images/model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: images/model_comparison.png")

# ============================================================
# 3. ROC CURVES
# ============================================================
print("Creating ROC curves chart...")
fig, ax = plt.subplots(figsize=(8, 6))

model_data = [
    ('Logistic Regression', lr_model.predict_proba(X_test)[:, 1], '#3498db'),
    ('Random Forest', rf_model.predict_proba(X_test)[:, 1], '#2ecc71'),
    ('XGBoost', xgb_model.predict_proba(X_test)[:, 1], '#e74c3c')
]

for name, prob, color in model_data:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, linewidth=2.5)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5, alpha=0.7)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/roc_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: images/roc_curves.png")

# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================
print("Creating feature importance chart...")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
bars = ax.barh(feature_importance['feature'], feature_importance['importance'],
               color=colors, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 10 Features for Predicting Churn', fontsize=14, fontweight='bold', pad=15)

# Add value labels
for bar, val in zip(bars, feature_importance['importance']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)

ax.set_xlim(0, feature_importance['importance'].max() * 1.15)
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: images/feature_importance.png")

# ============================================================
# 5. CHURN BY CONTRACT TYPE
# ============================================================
print("Creating churn by contract type chart...")
contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
contract_churn = contract_churn.reindex(['Month-to-month', 'One year', 'Two year'])

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#e74c3c', '#f39c12', '#2ecc71']
bars = ax.bar(contract_churn.index, contract_churn.values, color=colors,
              edgecolor='black', linewidth=1.2)

# Add value labels
for bar, val in zip(bars, contract_churn.values):
    ax.annotate(f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Churn Rate (%)', fontsize=12)
ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, contract_churn.max() * 1.15)
ax.axhline(y=26.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Overall Avg (26.5%)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('images/churn_by_contract.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: images/churn_by_contract.png")

print("\nAll visualizations generated successfully!")
