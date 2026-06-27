import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os

os.makedirs('outputs', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
demo   = pd.read_csv('demographics.csv')
labels = pd.read_csv('disease_risk_labels.csv')
vitals = pd.read_csv('vitals_time_series.csv')

print("Files loaded successfully.")
print(f"Demographics: {demo.shape}")
print(f"Labels:       {labels.shape}")
print(f"Vitals:       {vitals.shape}")

# ── 2. TREND FEATURES (computed on raw vitals BEFORE aggregation) ─────────────
print("\nComputing trend features...")

def compute_trends(group):
    group = group.reset_index(drop=True)
    n     = len(group)
    mid   = n // 2

    first_half  = group.iloc[:mid]
    second_half = group.iloc[mid:]

    hr_trend   = second_half['heart_rate'].mean()  - first_half['heart_rate'].mean()
    bp_trend   = second_half['systolic_bp'].mean() - first_half['systolic_bp'].mean()
    spo2_trend = second_half['spo2'].mean()        - first_half['spo2'].mean()

    return pd.Series({
        'heart_rate_trend':  hr_trend,
        'systolic_bp_trend': bp_trend,
        'spo2_trend':        spo2_trend
    })

trends = vitals.groupby('patient_id').apply(
    compute_trends, include_groups=False
).reset_index()

print(f"Trends computed: {trends.shape}")
print(trends.head(3))

# ── 3. AGGREGATE VITALS ───────────────────────────────────────────────────────
print("\nAggregating vitals...")

vitals_agg = vitals.groupby('patient_id').agg(
    heart_rate_mean    = ('heart_rate',   'mean'),
    heart_rate_max     = ('heart_rate',   'max'),
    heart_rate_min     = ('heart_rate',   'min'),
    heart_rate_std     = ('heart_rate',   'std'),
    systolic_bp_mean   = ('systolic_bp',  'mean'),
    systolic_bp_max    = ('systolic_bp',  'max'),
    diastolic_bp_mean  = ('diastolic_bp', 'mean'),
    diastolic_bp_std   = ('diastolic_bp', 'std'),
    temperature_mean   = ('temperature',  'mean'),
    temperature_max    = ('temperature',  'max'),
    spo2_mean          = ('spo2',         'mean'),
    spo2_min           = ('spo2',         'min'),
    spo2_std           = ('spo2',         'std'),
).reset_index()

# merge trends into aggregated vitals
vitals_agg = vitals_agg.merge(trends, on='patient_id')
print(f"Vitals aggregated with trends: {vitals_agg.shape}")

# ── 4. CLINICAL FEATURE ENGINEERING ──────────────────────────────────────────
print("\nEngineering clinical features...")

# Pulse Pressure — measures arterial stiffness
vitals_agg['pulse_pressure'] = (
    vitals_agg['systolic_bp_max'] - vitals_agg['diastolic_bp_mean']
)

# Mean Arterial Pressure — overall organ blood flow indicator
vitals_agg['mean_arterial_pressure'] = (
    (vitals_agg['systolic_bp_mean'] + 2 * vitals_agg['diastolic_bp_mean']) / 3
)

# Shock Index — heart rate / systolic BP, flags cardiovascular stress
vitals_agg['shock_index'] = (
    vitals_agg['heart_rate_mean'] / vitals_agg['systolic_bp_mean']
)

# SpO2 Danger Flag — 1 if SpO2 ever dropped below 94
vitals_agg['spo2_danger_flag'] = (
    vitals_agg['spo2_min'] < 94
).astype(int)

# Vitals Instability Score — how unstable readings are overall
vitals_agg['instability_score'] = (
    vitals_agg['heart_rate_std'] +
    vitals_agg['diastolic_bp_std'] +
    vitals_agg['spo2_std']
)

print("Clinical features added:")
print("  pulse_pressure, mean_arterial_pressure, shock_index,")
print("  spo2_danger_flag, instability_score")

# ── 5. MERGE ALL ──────────────────────────────────────────────────────────────
df = demo.merge(vitals_agg, on='patient_id').merge(labels, on='patient_id')
print(f"\nMerged dataset: {df.shape}")

# ── 6. ENCODE CATEGORICALS ────────────────────────────────────────────────────
le_gender   = LabelEncoder()
le_smoking  = LabelEncoder()
le_diabetes = LabelEncoder()
le_hyper    = LabelEncoder()

df['gender']         = le_gender.fit_transform(df['gender'].str.strip())
df['smoking_status'] = le_smoking.fit_transform(df['smoking_status'].str.strip())
df['diabetes']       = le_diabetes.fit_transform(df['diabetes'].str.strip())
df['hypertension']   = le_hyper.fit_transform(df['hypertension'].str.strip())

print("\nGender encoding:",      dict(zip(le_gender.classes_,   le_gender.transform(le_gender.classes_))))
print("Smoking encoding:",       dict(zip(le_smoking.classes_,  le_smoking.transform(le_smoking.classes_))))
print("Diabetes encoding:",      dict(zip(le_diabetes.classes_, le_diabetes.transform(le_diabetes.classes_))))
print("Hypertension encoding:",  dict(zip(le_hyper.classes_,    le_hyper.transform(le_hyper.classes_))))

le_target = LabelEncoder()
df['risk_encoded'] = le_target.fit_transform(df['risk_level'].str.strip())
print("\nTarget classes:",  le_target.classes_)
print("Target encoding:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

# ── 7. FEATURES & TARGET ──────────────────────────────────────────────────────
feature_cols = [
    # demographics
    'age', 'gender', 'smoking_status', 'diabetes', 'hypertension',
    # vitals aggregations
    'heart_rate_mean', 'heart_rate_max', 'heart_rate_min', 'heart_rate_std',
    'systolic_bp_mean', 'systolic_bp_max',
    'diastolic_bp_mean', 'diastolic_bp_std',
    'temperature_mean', 'temperature_max',
    'spo2_mean', 'spo2_min', 'spo2_std',
    # trend features
    'heart_rate_trend', 'systolic_bp_trend', 'spo2_trend',
    # clinical engineered features
    'pulse_pressure', 'mean_arterial_pressure',
    'shock_index', 'spo2_danger_flag', 'instability_score'
]

X = df[feature_cols]
y = df['risk_encoded']

print(f"\nTotal features: {len(feature_cols)}")
print("Feature list:", feature_cols)
print("\nClass distribution before SMOTE:", y.value_counts().to_dict())

# ── 8. SMOTE ──────────────────────────────────────────────────────────────────
sm = SMOTE(random_state=42, k_neighbors=2)
X_res, y_res = sm.fit_resample(X, y)
print("Class distribution after SMOTE: ", pd.Series(y_res).value_counts().to_dict())

# ── 9. TRAIN TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res,
    test_size=0.2,
    random_state=42,
    stratify=y_res
)

print(f"\nTrain size: {X_train.shape}")
print(f"Test size:  {X_test.shape}")

# ── 10. TRAIN MODEL ───────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=10,
    min_samples_leaf=2,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete.")

# ── 11. EVALUATE ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n── CLASSIFICATION REPORT ──")
print(classification_report(
    y_test, y_pred,
    target_names=le_target.classes_,
    zero_division=0
))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='f1_weighted')
print(f"Cross-val F1 (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"Previous Cross-val F1: 0.739")

if cv_scores.mean() > 0.739:
    print("IMPROVEMENT: New features helped.")
else:
    print("NO IMPROVEMENT: Consider reverting new features.")

# ── 12. FEATURE IMPORTANCE ────────────────────────────────────────────────────
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\n── TOP 15 FEATURES ──")
print(importance.head(15))

plt.figure(figsize=(12, 7))
importance.head(15).plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Top 15 Feature Importances — HealthGuard AI', fontsize=14)
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.close()
print("Feature importance chart saved.")

# ── 13. CONFUSION MATRIX ──────────────────────────────────────────────────────
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_,
            cmap='Blues')
plt.title('Confusion Matrix — HealthGuard AI')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved.")

# ── 14. SHAP ────────────────────────────────────────────────────────────────
print("\nGenerating SHAP summary...")

explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer(X_test)

print(type(shap_values))
print(shap_values.values.shape)
print(shap_values.base_values.shape)
print(shap_values.data.shape)

# Plot SHAP summary for HIGH RISK class
high_class = 0   # High = 0 according to your LabelEncoder

summary_exp = shap.Explanation(
    values=shap_values.values[:, :, high_class],
    base_values=shap_values.base_values[:, high_class],
    data=X_test.values,
    feature_names=feature_cols
)

plt.figure(figsize=(12,7))

shap.summary_plot(
    summary_exp.values,
    summary_exp.data,
    feature_names=summary_exp.feature_names,
    show=False
)

plt.tight_layout()
plt.savefig(
    "outputs/shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("SHAP summary saved.")

# ── 15. SAVE EVERYTHING ───────────────────────────────────────────────────────
try:
    print("\nSaving all outputs...")

    joblib.dump(model,        'outputs/healthguard_model.pkl')
    print("Model saved.")

    joblib.dump(le_target,    'outputs/label_encoder.pkl')
    print("Label encoder saved.")

    joblib.dump(explainer,    'outputs/shap_explainer.pkl')
    print("SHAP explainer saved.")

    joblib.dump(feature_cols, 'outputs/feature_cols.pkl')
    print("Feature cols saved.")

    encoders = {
        'gender':       le_gender,
        'smoking':      le_smoking,
        'diabetes':     le_diabetes,
        'hypertension': le_hyper
    }
    joblib.dump(encoders,     'outputs/encoders.pkl')
    print("Encoders saved.")

    df.to_csv('outputs/merged_clean.csv', index=False)
    print("Merged CSV saved.")

    print("\n── ALL OUTPUTS SAVED ──")
    print("outputs/healthguard_model.pkl")
    print("outputs/label_encoder.pkl")
    print("outputs/shap_explainer.pkl")
    print("outputs/feature_cols.pkl")
    print("outputs/encoders.pkl")
    print("outputs/merged_clean.csv")
    print("outputs/feature_importance.png")
    print("outputs/confusion_matrix.png")
    print("outputs/shap_summary.png")
    print("\nModel pipeline complete. Ready for Streamlit.")

except Exception as e:
    print(f"\nERROR DURING SAVE: {e}")
    import traceback
    traceback.print_exc()