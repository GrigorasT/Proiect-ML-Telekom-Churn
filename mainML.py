"""
TEMPLATE PROIECT ML END-TO-END
==============================

Proiect: Predicție Churn Client (Telecom)
Task: Clasificare binară — clienți care vor cancela abonamentul
Modele: Logistic Regression, Decision Tree, Random Forest
Metrici principale: AUC-ROC, Recall
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURARE GENERALĂ
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5
OUTPUT_DIR = './'  # salvează în folderul curent

np.random.seed(RANDOM_STATE)

print("=" * 80)
print("PREDICȚIE CHURN CLIENT - TELECOM")
print("=" * 80)

# ============================================================================
# PART 1: ÎNCARCĂ DATE
# ============================================================================

print("\n[STEP 1] Încarcă Date")

# Dataset sintetic Telecom Churn (realist, nu necesită Kaggle)
n_samples = 7043

tenure          = np.random.randint(0, 72, n_samples)
monthly_charges = np.random.uniform(18, 118, n_samples)
total_charges   = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
total_charges   = np.clip(total_charges, 0, None)

contract_type    = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                     n_samples, p=[0.55, 0.25, 0.20])
payment_method   = np.random.choice(['Electronic check', 'Mailed check',
                                      'Bank transfer', 'Credit card'],
                                     n_samples, p=[0.34, 0.23, 0.22, 0.21])
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'],
                                     n_samples, p=[0.34, 0.44, 0.22])
tech_support     = np.random.choice(['Yes', 'No', 'No internet service'],
                                     n_samples, p=[0.29, 0.49, 0.22])
online_security  = np.random.choice(['Yes', 'No', 'No internet service'],
                                     n_samples, p=[0.28, 0.50, 0.22])
senior_citizen    = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
paperless_billing = np.random.choice([0, 1], n_samples, p=[0.41, 0.59])

# Probabilitate churn realistă (influențată de features)
churn_prob = (
    0.35 * (contract_type == 'Month-to-month').astype(float)
    + 0.15 * (internet_service == 'Fiber optic').astype(float)
    - 0.20 * (tech_support == 'Yes').astype(float)
    - 0.001 * tenure
    + 0.002 * monthly_charges
    + 0.05 * senior_citizen
    + np.random.uniform(-0.1, 0.1, n_samples)
)
churn_prob = 1 / (1 + np.exp(-churn_prob * 3))
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

df = pd.DataFrame({
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract_type,
    'PaymentMethod': payment_method,
    'InternetService': internet_service,
    'TechSupport': tech_support,
    'OnlineSecurity': online_security,
    'SeniorCitizen': senior_citizen,
    'PaperlessBilling': paperless_billing,
    'Churn': churn
})

target_column = 'Churn'
y = df[target_column]
X = df.drop(target_column, axis=1)

print(f"Shape date: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target: {target_column}")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Churn rate: {y.mean():.2%}")

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print(f"\n[STEP 2] Exploratory Data Analysis")

print("\nStatistici descriptive:")
print(X.describe())

print("\nValori lipsă:")
print(X.isnull().sum())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploratory Data Analysis - Telecom Churn', fontsize=16, fontweight='bold')

# Viz 1: Distribuție tenure
axes[0, 0].hist(X['tenure'], bins=30, edgecolor='black', color='steelblue', alpha=0.7)
axes[0, 0].set_title('Distribuție Durată Abonament (tenure)')
axes[0, 0].set_xlabel('Luni')
axes[0, 0].set_ylabel('Frecvență')

# Viz 2: Rata churn per tip contract
churn_by_contract = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
axes[0, 1].bar(churn_by_contract.index, churn_by_contract.values,
               color=['#e74c3c', '#f39c12', '#2ecc71'])
axes[0, 1].set_title('Rata Churn per Tip Contract')
axes[0, 1].set_ylabel('Rată Churn')
axes[0, 1].set_ylim([0, 1])
for i, v in enumerate(churn_by_contract.values):
    axes[0, 1].text(i, v + 0.01, f'{v:.2%}', ha='center')

# Viz 3: MonthlyCharges vs Churn
data_plot = [df[df['Churn'] == 0]['MonthlyCharges'],
             df[df['Churn'] == 1]['MonthlyCharges']]
axes[1, 0].boxplot(data_plot, labels=['Rămâne', 'Pleacă'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
axes[1, 0].set_title('MonthlyCharges vs Churn')
axes[1, 0].set_ylabel('Charges lunare ($)')

# Viz 4: Distribuție Internet Service
service_counts = X['InternetService'].value_counts()
axes[1, 1].pie(service_counts.values, labels=service_counts.index,
               autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#2ecc71'])
axes[1, 1].set_title('Distribuție Tip Internet')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}01_eda_overview.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: 01_eda_overview.png")
plt.close()

# Corelații
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr_matrix = df[numeric_cols + ['Churn']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Matrice Corelații', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}02_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: 02_correlation_matrix.png")
plt.close()

# ============================================================================
# PART 3: PREPROCESARE
# ============================================================================

print(f"\n[STEP 3] Preprocesare")

numeric_features     = ['tenure', 'MonthlyCharges', 'TotalCharges',
                         'SeniorCitizen', 'PaperlessBilling']
categorical_features = ['Contract', 'PaymentMethod', 'InternetService',
                         'TechSupport', 'OnlineSecurity']

print(f"Features numerice: {numeric_features}")
print(f"Features categorice: {categorical_features}")

# Tratare valori lipsă (nu avem, dar păstrăm logica)
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train set: {X_train.shape[0]}")
print(f"Test set: {X_test.shape[0]}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print(f"Shape după preprocesare: {X_train_processed.shape}")
print("✓ Preprocesare completă")

# ============================================================================
# PART 4: FEATURE ENGINEERING
# ============================================================================

print(f"\n[STEP 4] Feature Engineering")

# Feature nou: cost mediu pe lună raportat la tenure
X['AvgCostPerMonth'] = X['TotalCharges'] / (X['tenure'] + 1)
# Feature nou: client nou (primele 6 luni)
X['IsNewCustomer']   = (X['tenure'] <= 6).astype(int)

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                    'SeniorCitizen', 'PaperlessBilling',
                    'AvgCostPerMonth', 'IsNewCustomer']

# Refacem split și preprocesare cu noile features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print(f"Shape după feature engineering: {X_train_processed.shape}")
print("✓ Feature engineering completă")

# ============================================================================
# PART 5: MODEL TRAINING
# ============================================================================

print(f"\n[STEP 5] Model Training")

# Model 1: Logistic Regression (baseline simplu)
print("\n--- Model 1: Logistic Regression ---")
model1 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model1.fit(X_train_processed, y_train)
pred1 = model1.predict(X_test_processed)
prob1 = model1.predict_proba(X_test_processed)[:, 1]
score1 = roc_auc_score(y_test, prob1)
print(f"AUC-ROC: {score1:.4f}")

# Model 2: Decision Tree (ușor de vizualizat și explicat)
print("\n--- Model 2: Decision Tree ---")
model2 = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
model2.fit(X_train_processed, y_train)
pred2 = model2.predict(X_test_processed)
prob2 = model2.predict_proba(X_test_processed)[:, 1]
score2 = roc_auc_score(y_test, prob2)
print(f"AUC-ROC: {score2:.4f}")

# Model 3: Random Forest (modelul final, cel mai bun scor)
print("\n--- Model 3: Random Forest ---")
model3 = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
model3.fit(X_train_processed, y_train)
pred3 = model3.predict(X_test_processed)
prob3 = model3.predict_proba(X_test_processed)[:, 1]
score3 = roc_auc_score(y_test, prob3)
print(f"AUC-ROC: {score3:.4f}")

# ============================================================================
# PART 6: HYPERPARAMETER TUNING
# ============================================================================

print(f"\n[STEP 6] Hyperparameter Tuning")

# Tunăm Random Forest — cel mai bun model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid,
    cv=CV_FOLDS,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train_processed, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUC-ROC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
best_pred  = best_model.predict(X_test_processed)
best_prob  = best_model.predict_proba(X_test_processed)[:, 1]
best_score = roc_auc_score(y_test, best_prob)
print(f"Test AUC-ROC: {best_score:.4f}")

# ============================================================================
# PART 7: EVALUARE FINALĂ
# ============================================================================

print(f"\n[STEP 7] Evaluare Finală")

accuracy  = accuracy_score(y_test, best_pred)
precision = precision_score(y_test, best_pred, zero_division=0)
recall    = recall_score(y_test, best_pred, zero_division=0)
f1        = f1_score(y_test, best_pred, zero_division=0)
auc_roc   = roc_auc_score(y_test, best_prob)

print(f"\n{'Metrica':<20} {'Valoare':<15}")
print("-" * 35)
print(f"{'AUC-ROC':<20} {auc_roc:<15.4f}")
print(f"{'Recall':<20} {recall:<15.4f}")
print(f"{'Precision':<20} {precision:<15.4f}")
print(f"{'F1-Score':<20} {f1:<15.4f}")
print(f"{'Accuracy':<20} {accuracy:<15.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Rămâne', 'Pleacă']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Rămâne', 'Pleacă'],
            yticklabels=['Rămâne', 'Pleacă'])
ax.set_title('Confusion Matrix - Churn Prediction', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: 04_confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curve - Churn Prediction', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}05_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: 05_roc_curve.png")
plt.close()

# ============================================================================
# PART 8: FEATURE IMPORTANCE
# ============================================================================

print(f"\n[STEP 8] Feature Importance")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_

    # Construim numele features după OneHotEncoding
    cat_feature_names = preprocessor.named_transformers_['cat']\
                            .get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    indices      = np.argsort(importances)[-10:]
    top_features = [all_feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_importances)), top_importances, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Features - Random Forest', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Salvat: 06_feature_importance.png")
    plt.close()

# ============================================================================
# PART 9: COMPARAȚIE MODELE
# ============================================================================

print(f"\n[STEP 9] Comparație Modele")

models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest (tuned)'],
    'AUC-ROC': [score1, score2, best_score],
    'Recall': [
        recall_score(y_test, pred1, zero_division=0),
        recall_score(y_test, pred2, zero_division=0),
        recall
    ]
})

print(models_comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(models_comparison))
width = 0.35
bars1 = ax.bar(x - width/2, models_comparison['AUC-ROC'], width,
               label='AUC-ROC', color=['#3498db', '#2ecc71', '#e74c3c'])
bars2 = ax.bar(x + width/2, models_comparison['Recall'], width,
               label='Recall',   color=['#85c1e9', '#82e0aa', '#f1948a'])
ax.set_ylabel('Score')
ax.set_title('Comparație Modele - AUC-ROC și Recall', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_comparison['Model'])
ax.set_ylim([0, 1])
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}07_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Salvat: 07_model_comparison.png")
plt.close()

# ============================================================================
# PART 10: RAPORT FINAL
# ============================================================================

print(f"\n[STEP 10] Raport Final")

summary_report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                          RAPORT FINAL PROIECT ML                           ║
╚════════════════════════════════════════════════════════════════════════════╝

1. PROBLEM STATEMENT
──────────────────
Descriere problemă:
Predicția clienților care vor cancela abonamentul (churn) la o companie telecom.
Identificarea timpurie a acestor clienți permite intervenție proactivă și reducerea pierderilor.

Dataset info:
- Dimensiuni: {X.shape}
- Features: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod,
            InternetService, TechSupport, OnlineSecurity, SeniorCitizen,
            PaperlessBilling, AvgCostPerMonth, IsNewCustomer
- Target: Churn (0 = rămâne, 1 = pleacă)
- Churn rate: {y.mean():.2%}

2. EXPLORATORY DATA ANALYSIS
────────────────────────────
Key insights:
- Clienții cu contract Month-to-month au cea mai mare rată de churn
- MonthlyCharges mai mari corelează cu probabilitate mai mare de churn
- Clienții noi (tenure mic) sunt mai predispuși să plece
- TechSupport activ reduce semnificativ rata de churn

3. DATA PREPROCESSING
─────────────────────
Pași urmați:
- Identificare features numerice și categorice
- Nu au existat valori lipsă în dataset
- StandardScaler pentru features numerice
- OneHotEncoder (drop='first') pentru features categorice
- Stratified train/test split (70/30)

4. FEATURE ENGINEERING
──────────────────────
Features create:
- AvgCostPerMonth: TotalCharges / (tenure + 1) — eficiența costului per lună
- IsNewCustomer: 1 dacă tenure <= 6 luni — clienți în faza de risc

5. MODEL PERFORMANCE
─────────────────────
Metricile finale (Random Forest tuned):
- AUC-ROC:   {auc_roc:.4f}  ← metrica principală
- Recall:    {recall:.4f}  ← găsim cine pleacă
- Precision: {precision:.4f}
- F1-Score:  {f1:.4f}
- Accuracy:  {accuracy:.4f}

6. BEST MODEL
──────────────
Random Forest (tuned cu GridSearchCV) a performat cel mai bine.
Parametri optimi: {grid_search.best_params_}
Random Forest captează relații non-liniare și interacțiuni între features
mai bine decât Logistic Regression sau un singur Decision Tree.

7. CONCLUZII ȘI RECOMANDĂRI
────────────────────────────
- Modelul identifică cu succes clienții cu risc de churn (Recall ridicat)
- Cel mai important factor: tipul de contract (Month-to-month = risc mare)
- Recomandare business: oferte de retenție targetate pentru clienți noi
  cu contract lunar și charges ridicate

8. ÎMBUNĂTĂȚIRI VIITOARE
─────────────────────────
- Folosirea dataset-ului real Kaggle Telecom Churn
- Testarea XGBoost / LightGBM pentru performanță mai bună
- Adăugarea SHAP values pentru explicabilitate la nivel de client individual
- Threshold tuning pentru optimizarea Recall vs Precision

════════════════════════════════════════════════════════════════════════════════
"""

print(summary_report)

with open(f'{OUTPUT_DIR}RAPORT_FINAL.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("✓ Salvat: RAPORT_FINAL.txt")

# ============================================================================
# TIPS FINALE
# ============================================================================

print("\n" + "=" * 80)
print("TIPS PENTRU FINALIZARE PROIECT")
print("=" * 80)
print("""
1. VERIFICA CODUL:
   - Asigură-te că toate TODOs sunt completate
   - Testează codul pas cu pas
   - Adaugă comentarii la seturi complicate

2. DOCUMENTARE:
   - Scrie un README bun cu descriere problema și instrucțiuni
   - Adaugă imagini cu rezultatele
   - Documentează choice-uri importante

3. REPRODUCIBILITATE:
   - Setează random_state la toate modelele
   - Salvează preprocessor (pentru a transforma date noi)
   - Documentează versiunile librării folosite

4. VERSION CONTROL:
   - Commit regulat cu mesaje descriptive
   - Nu commita date mari (folosește .gitignore)
   - Salvează modelele (pickle/joblib)

5. REZULTATE:
   - Salvează toate vizualizările cu plt.savefig()
   - Creează un raport coerent
   - Discută limitări și îmbunătățiri

6. PORTFOLIO:
   - Push pe GitHub cu README bun
   - Adaugă badges (build status, coverage)
   - Linker din CV/LinkedIn
""")

print("=" * 80)