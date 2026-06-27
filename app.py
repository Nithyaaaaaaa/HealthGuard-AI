import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide"
)

# ── LOAD MODEL & DATA ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load('outputs/healthguard_model.pkl')
    le_target    = joblib.load('outputs/label_encoder.pkl')
    explainer    = joblib.load('outputs/shap_explainer.pkl')
    feature_cols = joblib.load('outputs/feature_cols.pkl')
    encoders     = joblib.load('outputs/encoders.pkl')
    return model, le_target, explainer, feature_cols, encoders

@st.cache_data
def load_data():
    demo   = pd.read_csv('demographics.csv')
    labels = pd.read_csv('disease_risk_labels.csv')
    vitals = pd.read_csv('vitals_time_series.csv')
    merged = pd.read_csv('outputs/merged_clean.csv')
    df     = demo.merge(labels, on='patient_id')
    return demo, labels, vitals, merged, df

model, le_target, explainer, feature_cols, encoders = load_model()
demo, labels, vitals, merged, df = load_data()

COLORS = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

# ════════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS — used by Risk Prediction page AND Patient Lookup page
# ════════════════════════════════════════════════════════════════════════════════

def build_engineered_features(heart_rate, systolic_bp, diastolic_bp, temperature, spo2,
                               heart_rate_trend=0.0, systolic_bp_trend=0.0, spo2_trend=0.0):
    """
    Derives the 5 new clinical engineered features + std/min/max approximations
    from raw vital inputs, the same way model.py derives them from raw vitals
    time series. Used for the manual slider-based prediction form, where we
    only have single point-in-time values instead of 24 readings per patient.
    """
    heart_rate_max   = heart_rate * 1.10
    heart_rate_min   = heart_rate * 0.90
    heart_rate_std   = heart_rate * 0.05
    systolic_bp_max  = systolic_bp * 1.10
    diastolic_bp_std = diastolic_bp * 0.05
    temperature_max  = temperature + 0.3
    spo2_min         = spo2 - 1.5
    spo2_std         = 0.5

    pulse_pressure          = systolic_bp_max - diastolic_bp
    mean_arterial_pressure  = (systolic_bp + 2 * diastolic_bp) / 3
    shock_index             = heart_rate / systolic_bp
    spo2_danger_flag        = int(spo2_min < 94)
    instability_score       = heart_rate_std + diastolic_bp_std + spo2_std

    return {
        'heart_rate_max':   heart_rate_max,
        'heart_rate_min':   heart_rate_min,
        'heart_rate_std':   heart_rate_std,
        'systolic_bp_max':  systolic_bp_max,
        'diastolic_bp_std': diastolic_bp_std,
        'temperature_max':  temperature_max,
        'spo2_min':         spo2_min,
        'spo2_std':         spo2_std,
        'heart_rate_trend':  heart_rate_trend,
        'systolic_bp_trend': systolic_bp_trend,
        'spo2_trend':        spo2_trend,
        'pulse_pressure':            pulse_pressure,
        'mean_arterial_pressure':    mean_arterial_pressure,
        'shock_index':               shock_index,
        'spo2_danger_flag':          spo2_danger_flag,
        'instability_score':         instability_score,
    }


def get_clinical_breakdown(age, gender, smoking, diabetes, hypertension,
                            heart_rate, systolic_bp, diastolic_bp, temperature, spo2,
                            pulse_pressure=None, mean_arterial_pressure=None,
                            shock_index=None, spo2_danger_flag=None, instability_score=None,
                            heart_rate_trend=None, systolic_bp_trend=None, spo2_trend=None):
    """
    Single shared rule-engine. Returns (issues, warnings, healthy) lists of
    human-readable strings, each with normal range + target range, so the
    Risk Prediction page and the Patient Lookup page never drift out of sync.
    smoking / diabetes / hypertension are expected as plain strings:
    smoking in {"Never","Former","Current"}, diabetes/hypertension in {"No","Yes"}.
    """
    issues   = []
    warnings = []
    healthy  = []

    # ── Core vitals (existing logic, unchanged) ──────────────────────────────
    if systolic_bp >= 140:
        issues.append(f"**Systolic BP is high ({systolic_bp:.0f} mmHg)** — "
                      f"Normal: 90–120 mmHg. Target: bring below 120 mmHg via "
                      f"medication or lifestyle changes.")
    elif systolic_bp >= 120:
        warnings.append(f"**Systolic BP is elevated ({systolic_bp:.0f} mmHg)** — "
                        f"Borderline high. Target: below 120 mmHg.")
    else:
        healthy.append(f"Systolic BP is normal ({systolic_bp:.0f} mmHg)")

    if diastolic_bp >= 90:
        issues.append(f"**Diastolic BP is high ({diastolic_bp:.0f} mmHg)** — "
                      f"Normal: 60–80 mmHg. Target: bring below 80 mmHg.")
    elif diastolic_bp >= 80:
        warnings.append(f"**Diastolic BP is borderline ({diastolic_bp:.0f} mmHg)** — "
                        f"Target: below 80 mmHg.")
    else:
        healthy.append(f"Diastolic BP is normal ({diastolic_bp:.0f} mmHg)")

    if spo2 < 90:
        issues.append(f"**SpO2 is critically low ({spo2:.0f}%)** — "
                      f"Normal: 95–100%. Target: immediate oxygen supplementation, "
                      f"bring above 95%.")
    elif spo2 < 95:
        issues.append(f"**SpO2 is low ({spo2:.0f}%)** — "
                      f"Normal: 95–100%. Target: above 95%, monitor closely.")
    else:
        healthy.append(f"SpO2 is normal ({spo2:.0f}%)")

    if heart_rate > 100:
        warnings.append(f"**Heart rate is elevated ({heart_rate:.0f} bpm)** — "
                        f"Normal: 60–100 bpm. Target: bring below 100 bpm; "
                        f"investigate stress, fever, or cardiac issues.")
    elif heart_rate < 60:
        warnings.append(f"**Heart rate is low ({heart_rate:.0f} bpm)** — "
                        f"Normal: 60–100 bpm. Target: bring above 60 bpm; "
                        f"monitor for bradycardia.")
    else:
        healthy.append(f"Heart rate is normal ({heart_rate:.0f} bpm)")

    if temperature >= 38.0:
        issues.append(f"**Temperature indicates fever ({temperature:.1f}°C)** — "
                      f"Normal: 36.1–37.2°C. Target: below 37.2°C; investigate infection.")
    elif temperature >= 37.3:
        warnings.append(f"**Temperature is slightly elevated ({temperature:.1f}°C)** — "
                        f"Target: below 37.2°C; monitor for fever development.")
    elif temperature < 36.0:
        warnings.append(f"**Temperature is low ({temperature:.1f}°C)** — "
                        f"Target: above 36.1°C; monitor for hypothermia.")
    else:
        healthy.append(f"Temperature is normal ({temperature:.1f}°C)")

    if age >= 65:
        warnings.append(f"**Age is a risk factor ({age} years)** — "
                        f"Patients above 65 require closer monitoring.")

    if smoking == "Current":
        warnings.append("**Active smoker** — Significantly increases cardiovascular risk. "
                        "Target: enroll in smoking cessation program.")
    if diabetes == "Yes":
        warnings.append("**Diabetes present** — Target: HbA1c below 7%, "
                        "fasting blood glucose 70–130 mg/dL.")
    if hypertension == "Yes" and systolic_bp >= 130:
        issues.append("**Hypertension with elevated BP** — "
                      "Current BP control is insufficient. Target: medication review "
                      "to bring systolic below 130 mmHg.")

    # ── New clinical engineered features ─────────────────────────────────────
    if pulse_pressure is not None:
        if pulse_pressure > 60:
            issues.append(f"**Pulse pressure is high ({pulse_pressure:.0f} mmHg)** — "
                          f"Normal: 40–60 mmHg. Indicates arterial stiffness. "
                          f"Target: bring below 60 mmHg by controlling systolic BP.")
        elif pulse_pressure < 25:
            warnings.append(f"**Pulse pressure is low ({pulse_pressure:.0f} mmHg)** — "
                            f"Normal: 40–60 mmHg. May indicate reduced stroke volume. "
                            f"Target: bring above 25 mmHg.")
        else:
            healthy.append(f"Pulse pressure is normal ({pulse_pressure:.0f} mmHg)")

    if mean_arterial_pressure is not None:
        if mean_arterial_pressure > 100:
            issues.append(f"**Mean Arterial Pressure is high ({mean_arterial_pressure:.0f} mmHg)** — "
                          f"Normal: 70–100 mmHg. Indicates excess strain on organs. "
                          f"Target: bring below 100 mmHg.")
        elif mean_arterial_pressure < 65:
            issues.append(f"**Mean Arterial Pressure is low ({mean_arterial_pressure:.0f} mmHg)** — "
                          f"Normal: 70–100 mmHg. Risk of inadequate organ perfusion. "
                          f"Target: bring above 65 mmHg urgently.")
        else:
            healthy.append(f"Mean Arterial Pressure is normal ({mean_arterial_pressure:.0f} mmHg)")

    if shock_index is not None:
        if shock_index > 0.9:
            issues.append(f"**Shock Index is high ({shock_index:.2f})** — "
                          f"Normal: 0.5–0.7. Flags cardiovascular stress / possible shock. "
                          f"Target: bring below 0.7 by stabilizing heart rate and BP.")
        elif shock_index > 0.7:
            warnings.append(f"**Shock Index is borderline ({shock_index:.2f})** — "
                            f"Normal: 0.5–0.7. Target: below 0.7.")
        else:
            healthy.append(f"Shock Index is normal ({shock_index:.2f})")

    if spo2_danger_flag is not None and spo2_danger_flag == 1:
        issues.append("**SpO2 danger flag triggered** — "
                      "Oxygen saturation dropped below 94% at some point. "
                      "Target: eliminate desaturation episodes, keep SpO2 above 95% at all times.")
    elif spo2_danger_flag is not None:
        healthy.append("No SpO2 danger episodes recorded")

    if instability_score is not None:
        if instability_score > 8:
            issues.append(f"**Vitals instability score is high ({instability_score:.1f})** — "
                          f"Readings are fluctuating significantly across heart rate, "
                          f"diastolic BP, and SpO2. Target: bring below 5 through closer "
                          f"monitoring and stabilization of underlying cause.")
        elif instability_score > 5:
            warnings.append(f"**Vitals instability score is elevated ({instability_score:.1f})** — "
                            f"Target: below 5.")
        else:
            healthy.append(f"Vitals are stable (instability score {instability_score:.1f})")

    # ── Trend features (direction of change over the monitoring window) ─────
    if heart_rate_trend is not None and abs(heart_rate_trend) >= 5:
        direction = "rising" if heart_rate_trend > 0 else "falling"
        warnings.append(f"**Heart rate trend is {direction} ({heart_rate_trend:+.1f} bpm "
                        f"across the monitoring window)** — Target: stabilize trend near 0.")
    if systolic_bp_trend is not None and abs(systolic_bp_trend) >= 5:
        direction = "rising" if systolic_bp_trend > 0 else "falling"
        warnings.append(f"**Systolic BP trend is {direction} ({systolic_bp_trend:+.1f} mmHg "
                        f"across the monitoring window)** — Target: stabilize trend near 0.")
    if spo2_trend is not None and abs(spo2_trend) >= 2:
        direction = "rising" if spo2_trend > 0 else "falling"
        severity = issues if spo2_trend < 0 else warnings
        severity.append(f"**SpO2 trend is {direction} ({spo2_trend:+.1f}% across the "
                        f"monitoring window)** — Target: stabilize or improve trend toward 0 or positive.")

    return issues, warnings, healthy


def render_clinical_breakdown(issues, warnings, healthy):
    """Renders the issues/warnings/healthy lists with consistent styling."""
    if issues:
        st.markdown("#### ⚠️ Critical Concerns")
        for issue in issues:
            st.error(f"🔴 {issue}")

    if warnings:
        st.markdown("#### ⚡ Parameters to Monitor")
        for warning in warnings:
            st.warning(f"🟡 {warning}")

    if healthy:
        st.markdown("#### ✅ Normal Parameters")
        for h in healthy:
            st.success(f"🟢 {h}")

    st.markdown("---")
    st.info(f"**Summary:** {len(issues)} critical concern(s) and "
            f"{len(warnings)} parameter(s) to monitor.")


def render_recommendations(risk_label):
    st.subheader("📋 Clinical Recommendations")
    if risk_label == 'High':
        st.markdown("""
- 🚨 Immediate review by attending physician
- 📋 Order comprehensive blood panel
- 💊 Review current medications
- 🏥 Consider hospital admission if vitals deteriorate
- 📞 Notify on-call specialist
        """)
    elif risk_label == 'Medium':
        st.markdown("""
- 📅 Schedule follow-up appointment within 2 weeks
- 📊 Monitor blood pressure daily
- 🏃 Recommend lifestyle modification plan
- 💉 Review vaccination and screening schedule
- 📱 Set up remote vitals monitoring
        """)
    else:
        st.markdown("""
- ✅ Continue routine annual check-up schedule
- 🥗 Maintain healthy diet and regular exercise
- 📊 Monitor vitals monthly
- 🧪 Standard preventive screening only
        """)


def render_risk_badge(risk_label, prefix=""):
    if risk_label == 'High':
        st.error(f"⚠️ {prefix}RISK LEVEL: **{risk_label}**")
    elif risk_label == 'Medium':
        st.warning(f"⚡ {prefix}RISK LEVEL: **{risk_label}**")
    else:
        st.success(f"✅ {prefix}RISK LEVEL: **{risk_label}**")


def render_shap_waterfall(shap_values, pred_idx, base_value, data_row, feature_names):

    explanation = shap.Explanation(
        values=shap_values.values[0, :, pred_idx],
        base_values=shap_values.base_values[0, pred_idx],
        data=shap_values.data[0],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(10,5))

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close()

    st.caption(
        "🔴 Red bars push toward higher risk. "
        "🔵 Blue bars push toward lower risk. "
        "Longer bars mean stronger influence on the prediction."
    )


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("HealthGuard AI")
st.sidebar.markdown("*Explainable Clinical Decision Support*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "🔍 Risk Prediction",
    "🧑‍⚕️ Patient Lookup",
    "📊 Data Analytics",
    "📈 Model Performance"
])
st.sidebar.markdown("---")
st.sidebar.caption("IEEE DataPort Hackathon 2026")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("HealthGuard AI")
    st.subheader("Explainable Clinical Decision Support System for Early Risk Prediction")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients",   "50")
    col2.metric("Model Accuracy",   "78%")
    col3.metric("Weighted F1",      "0.78")
    col4.metric("High Risk Recall", "100%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            risk_counts.index,
            risk_counts.values,
            color=[COLORS.get(x, '#3498db') for x in risk_counts.index],
            edgecolor='white', linewidth=1.2
        )
        for bar, val in zip(bars, risk_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    str(val), ha='center', fontweight='bold', fontsize=12)
        ax.set_title('Patient Risk Distribution', fontsize=13)
        ax.set_ylabel('Number of Patients')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Age Distribution by Risk Level")
        fig, ax = plt.subplots(figsize=(6, 4))
        for risk, color in COLORS.items():
            subset = df[df['risk_level'] == risk]['age']
            if len(subset) > 0:
                ax.hist(subset, alpha=0.7, label=risk,
                        color=color, bins=8, edgecolor='white')
        ax.set_title('Age Distribution by Risk Level', fontsize=13)
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📌 Key Clinical Finding")
    st.info(
        "💡 **Vital signs are stronger predictors of patient risk than lifestyle factors.** "
        "Systolic blood pressure, body temperature, and SpO2 ranked as the top predictors — "
        "outperforming demographic factors like smoking status, diabetes, and hypertension. "
        "This suggests that continuous vital monitoring is more predictive of acute risk "
        "than static patient history alone."
    )

    st.markdown("---")
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Demographics Features", "5")
    col2.metric("Vitals Readings", "1,200")
    col3.metric("Engineered Features", "23")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTION (manual sliders, live what-if)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Risk Prediction":
    st.title("🔍 Patient Risk Prediction")
    st.markdown("Enter patient details below to predict their healthcare risk level.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Demographics")
        age          = st.slider("Age", 18, 100, 45)
        gender       = st.selectbox("Gender", ["Male", "Female"])
        smoking      = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        diabetes     = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])

    with col2:
        st.subheader("Vital Signs")
        heart_rate   = st.slider("Heart Rate (bpm)",        50,   130,  75)
        systolic_bp  = st.slider("Systolic BP (mmHg)",      90,   200,  120)
        diastolic_bp = st.slider("Diastolic BP (mmHg)",     60,   130,  80)
        temperature  = st.slider("Body Temperature (°C)",   35.0, 40.0, 36.6)
        spo2         = st.slider("SpO2 (%)",                85,   100,  98)

    # ── Auto-derived engineered features (read-only, shown to user) ──────────
    eng = build_engineered_features(heart_rate, systolic_bp, diastolic_bp, temperature, spo2)

    with st.expander("🧮 Auto-calculated clinical features (derived from vitals above)"):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pulse Pressure", f"{eng['pulse_pressure']:.0f} mmHg")
        c2.metric("Mean Arterial Pressure", f"{eng['mean_arterial_pressure']:.0f} mmHg")
        c3.metric("Shock Index", f"{eng['shock_index']:.2f}")
        c4.metric("SpO2 Danger Flag", "Yes" if eng['spo2_danger_flag'] else "No")
        c5.metric("Instability Score", f"{eng['instability_score']:.1f}")
        st.caption("These update automatically as you move the sliders above — "
                   "they're not independent inputs, since they're mathematically "
                   "derived from heart rate, BP, and SpO2.")

    st.markdown("---")

    if st.button("🔍 Predict Risk Level", type="primary", use_container_width=True):

        # ── ENCODE INPUT ──────────────────────────────────────────────────────
        gender_enc = encoders["gender"].transform([gender])[0]

        smoking_enc = encoders["smoking"].transform([smoking])[0]

        diabetes_enc = encoders["diabetes"].transform([diabetes])[0]

        hyper_enc = encoders["hypertension"].transform([hypertension])[0]

        input_data = pd.DataFrame([{
            'age':               age,
            'gender':            gender_enc,
            'smoking_status':    smoking_enc,
            'diabetes':          diabetes_enc,
            'hypertension':      hyper_enc,
            'heart_rate_mean':   heart_rate,
            'heart_rate_max':    eng['heart_rate_max'],
            'heart_rate_min':    eng['heart_rate_min'],
            'heart_rate_std':    eng['heart_rate_std'],
            'systolic_bp_mean':  systolic_bp,
            'systolic_bp_max':   eng['systolic_bp_max'],
            'diastolic_bp_mean': diastolic_bp,
            'diastolic_bp_std':  eng['diastolic_bp_std'],
            'temperature_mean':  temperature,
            'temperature_max':   eng['temperature_max'],
            'spo2_mean':         spo2,
            'spo2_min':          eng['spo2_min'],
            'spo2_std':          eng['spo2_std'],
            'heart_rate_trend':  eng['heart_rate_trend'],
            'systolic_bp_trend': eng['systolic_bp_trend'],
            'spo2_trend':        eng['spo2_trend'],
            'pulse_pressure':            eng['pulse_pressure'],
            'mean_arterial_pressure':    eng['mean_arterial_pressure'],
            'shock_index':               eng['shock_index'],
            'spo2_danger_flag':          eng['spo2_danger_flag'],
            'instability_score':         eng['instability_score'],
        }])[feature_cols]  # enforce exact column order the model expects

        prediction    = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        risk_label    = le_target.inverse_transform([prediction])[0]

        # ==============================
        # Clinical Safety Override
        # ==============================

        override_reason = None

        # Critical hypoxemia
        if eng['spo2_min'] <= 85:
            risk_label = "High"
            override_reason = "Critical oxygen desaturation (SpO₂ ≤ 85%)"

        # Shock
        elif eng['shock_index'] >= 1.0:
            risk_label = "High"
            override_reason = "Shock Index ≥ 1.0"

        # Severe hypotension
        elif eng['mean_arterial_pressure'] < 60:
            risk_label = "High"
            override_reason = "Mean Arterial Pressure < 60 mmHg"

        # High fever with low oxygen
        elif temperature >= 39 and eng['spo2_min'] <= 90:            
            risk_label = "High"
            override_reason = "High fever with hypoxemia"

        # ── RESULT ────────────────────────────────────────────────────────────
        st.markdown("### Prediction Result")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # AI prediction
            st.markdown("### Prediction")
            render_risk_badge(le_target.inverse_transform([prediction])[0])

            # Clinical safety override (only if triggered)
            if override_reason:
                st.markdown("### Final Clinical Decision")
                st.error(
                    f"⚠️ Escalated to **HIGH RISK**\n\n"
                    f"**Reason:** {override_reason}"
                )

                final_risk = "High"
            else:
                final_risk = le_target.inverse_transform([prediction])[0]

        # Recommendation based on final clinical decision
        if final_risk == "High":
            st.error("Immediate clinical attention recommended.")
        elif final_risk == "Medium":
            st.warning("Schedule follow-up within 2 weeks.")
        else:
            st.success("Continue routine monitoring.")      

        # ── CONFIDENCE ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({
            'Risk Level':  le_target.classes_,
            'Probability': probabilities
        })
        fig, ax = plt.subplots(figsize=(7, 3))
        bar_colors = [COLORS.get(r, '#3498db') for r in prob_df['Risk Level']]
        bars = ax.barh(prob_df['Risk Level'],
                       prob_df['Probability'],
                       color=bar_colors, edgecolor='white')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Confidence by Risk Class')
        for bar, val in zip(bars, prob_df['Probability']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        # ── SHAP EXPLANATION ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 SHAP Explanation — Why this prediction?")
        st.markdown("*How each factor pushed the risk prediction for this specific patient:*")

        shap_values = explainer(input_data)

        pred_idx = int(prediction)

        render_shap_waterfall(
            shap_values,
            pred_idx,
            shap_values.base_values[0, pred_idx],
            input_data.iloc[0].values,
            feature_cols
        )

        # ── CLINICAL EXPLANATION ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🩺 Clinical Explanation")
        st.markdown("*Parameter-by-parameter breakdown for this patient — what's driving "
                    "the risk, and what range it needs to come down to:*")

        issues, warnings, healthy = get_clinical_breakdown(
            age, gender, smoking, diabetes, hypertension,
            heart_rate, systolic_bp, diastolic_bp, temperature, spo2,
            pulse_pressure=eng['pulse_pressure'],
            mean_arterial_pressure=eng['mean_arterial_pressure'],
            shock_index=eng['shock_index'],
            spo2_danger_flag=eng['spo2_danger_flag'],
            instability_score=eng['instability_score'],
            heart_rate_trend=eng['heart_rate_trend'],
            systolic_bp_trend=eng['systolic_bp_trend'],
            spo2_trend=eng['spo2_trend'],
        )
        render_clinical_breakdown(issues, warnings, healthy)

        # ── RECOMMENDATIONS ───────────────────────────────────────────────────
        st.markdown("---")
        render_recommendations(final_risk)
        
# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PATIENT LOOKUP  (NEW: real patient, real features, real SHAP)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧑‍⚕️ Patient Lookup":
    st.title("🧑‍⚕️ Patient Lookup")
    st.markdown("Select any patient to see their predicted risk, the exact factors "
                "driving it, and the target range each factor needs to reach.")
    st.markdown("---")

    patient_list     = sorted(merged['patient_id'].unique().tolist())
    selected_patient = st.selectbox("Select Patient ID", patient_list)

    prow = merged[merged['patient_id'] == selected_patient].iloc[0]

    # Build the exact feature vector the model was trained on, straight from merged_clean.csv
    patient_features = pd.DataFrame([prow[feature_cols].to_dict()])[feature_cols]

    prediction    = model.predict(patient_features)[0]
    probabilities = model.predict_proba(patient_features)[0]
    risk_label    = le_target.inverse_transform([prediction])[0]
    actual_label  = prow['risk_level']

    # ── HEADER SUMMARY ─────────────────────────────────────────────────────────
    st.markdown(f"### Patient: `{selected_patient}`")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Age", f"{int(prow['age'])}")
    col2.metric("Gender", "Male" if prow['gender'] == 1 else "Female")
    col3.metric("Predicted Risk", risk_label)
    col4.metric("Actual Label (dataset)", actual_label,
                delta="Match" if risk_label == actual_label else "Mismatch",
                delta_color="normal" if risk_label == actual_label else "inverse")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_risk_badge(risk_label)
        if risk_label == 'High':
            st.error("Immediate clinical attention recommended.")
        elif risk_label == 'Medium':
            st.warning("Schedule follow-up within 2 weeks.")
        else:
            st.success("Continue routine monitoring.")

    # ── CONFIDENCE ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({
        'Risk Level':  le_target.classes_,
        'Probability': probabilities
    })
    fig, ax = plt.subplots(figsize=(7, 3))
    bar_colors = [COLORS.get(r, '#3498db') for r in prob_df['Risk Level']]
    bars = ax.barh(prob_df['Risk Level'], prob_df['Probability'],
                   color=bar_colors, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title(f'Prediction Confidence — {selected_patient}')
    for bar, val in zip(bars, prob_df['Probability']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── SHAP EXPLANATION (real model SHAP for this real patient) ───────────────
    st.markdown("---")
    st.subheader("🔬 SHAP Explanation — Why this patient is this risk")
    st.markdown("*How each factor pushed this patient's actual prediction:*")

    shap_values = explainer(patient_features)

    pred_idx = int(prediction)

    render_shap_waterfall(
        shap_values,
        pred_idx,
        shap_values.base_values[0, pred_idx],
        patient_features.iloc[0].values,
        feature_cols
    )

    # ── CLINICAL EXPLANATION: "High risk because of X, Y, Z — target range" ───
    st.markdown("---")
    st.subheader("🩺 Clinical Explanation")
    st.markdown(f"*Why **{selected_patient}** is **{risk_label}** risk, and what range "
                f"each contributing factor needs to come down to:*")

    smoking_label = encoders['smoking'].inverse_transform([int(prow['smoking_status'])])[0]
    diabetes_label = encoders['diabetes'].inverse_transform([int(prow['diabetes'])])[0]
    hyper_label = encoders['hypertension'].inverse_transform([int(prow['hypertension'])])[0]

    issues, warnings, healthy = get_clinical_breakdown(
        age=prow['age'], gender=prow['gender'], smoking=smoking_label,
        diabetes=diabetes_label, hypertension=hyper_label,
        heart_rate=prow['heart_rate_mean'], systolic_bp=prow['systolic_bp_mean'],
        diastolic_bp=prow['diastolic_bp_mean'], temperature=prow['temperature_mean'],
        spo2=prow['spo2_mean'],
        pulse_pressure=prow['pulse_pressure'],
        mean_arterial_pressure=prow['mean_arterial_pressure'],
        shock_index=prow['shock_index'],
        spo2_danger_flag=prow['spo2_danger_flag'],
        instability_score=prow['instability_score'],
        heart_rate_trend=prow['heart_rate_trend'],
        systolic_bp_trend=prow['systolic_bp_trend'],
        spo2_trend=prow['spo2_trend'],
    )
    render_clinical_breakdown(issues, warnings, healthy)

    # ── RECOMMENDATIONS ──────────────────────────────────────────────────────────
    st.markdown("---")
    render_recommendations(risk_label)

    # ── VITALS TIMELINE FOR THIS PATIENT ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 24-Reading Vitals Timeline")
    patient_vitals = vitals[vitals['patient_id'] == selected_patient].reset_index(drop=True)
    vital_choice   = st.selectbox(
        "Select Vital to View",
        ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2'],
        key="lookup_vital_choice"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(patient_vitals.index, patient_vitals[vital_choice],
            color=COLORS.get(risk_label, '#3498db'),
            linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Reading Number (0–23)')
    ax.set_ylabel(vital_choice.replace('_', ' ').title())
    ax.set_title(f'{vital_choice.replace("_", " ").title()} over 24 Readings — {selected_patient}')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── RAW ENGINEERED FEATURE VALUES (for transparency / audit) ────────────────
    with st.expander("🧮 View this patient's full engineered feature vector"):
        feat_display = prow[feature_cols].to_frame(name="Value")
        st.dataframe(feat_display, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analytics":
    st.title("📊 Data Analytics")
    st.markdown("---")

    vitals_agg = vitals.groupby('patient_id').agg(
        heart_rate_mean  = ('heart_rate',  'mean'),
        systolic_bp_mean = ('systolic_bp', 'mean'),
        diastolic_bp_mean= ('diastolic_bp','mean'),
        temperature_mean = ('temperature', 'mean'),
        spo2_mean        = ('spo2',        'mean'),
    ).reset_index()
    merged_viz = df.merge(vitals_agg, on='patient_id')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Systolic BP by Risk Level")
        fig, ax = plt.subplots(figsize=(6, 4))
        for risk, color in COLORS.items():
            subset = merged_viz[merged_viz['risk_level'] == risk]['systolic_bp_mean']
            ax.hist(subset, alpha=0.7, label=risk, color=color, bins=8, edgecolor='white')
        ax.set_xlabel('Systolic BP (mmHg)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_title('Systolic BP Distribution by Risk')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("SpO2 by Risk Level")
        fig, ax = plt.subplots(figsize=(6, 4))
        for risk, color in COLORS.items():
            subset = merged_viz[merged_viz['risk_level'] == risk]['spo2_mean']
            ax.hist(subset, alpha=0.7, label=risk, color=color, bins=8, edgecolor='white')
        ax.set_xlabel('SpO2 (%)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_title('SpO2 Distribution by Risk')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Heart Rate by Risk Level")
        fig, ax = plt.subplots(figsize=(6, 4))
        data_box    = [merged_viz[merged_viz['risk_level'] == r]['heart_rate_mean'].values
                       for r in ['Low', 'Medium', 'High']]
        bp = ax.boxplot(data_box, labels=['Low', 'Medium', 'High'], patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS['Low'], COLORS['Medium'], COLORS['High']]):
            patch.set_facecolor(color)
        ax.set_ylabel('Heart Rate (bpm)')
        ax.set_title('Heart Rate by Risk Level')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Temperature by Risk Level")
        fig, ax = plt.subplots(figsize=(6, 4))
        data_box = [merged_viz[merged_viz['risk_level'] == r]['temperature_mean'].values
                    for r in ['Low', 'Medium', 'High']]
        bp = ax.boxplot(data_box, labels=['Low', 'Medium', 'High'], patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS['Low'], COLORS['Medium'], COLORS['High']]):
            patch.set_facecolor(color)
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature by Risk Level')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Smoking Status vs Risk Level")
    crosstab = pd.crosstab(df['smoking_status'], df['risk_level'])
    fig, ax  = plt.subplots(figsize=(8, 4))
    crosstab.plot(kind='bar', ax=ax,
                  color=[COLORS['High'], COLORS['Low'], COLORS['Medium']],
                  edgecolor='white')
    ax.set_xlabel('Smoking Status')
    ax.set_ylabel('Count')
    ax.set_title('Smoking Status vs Risk Level')
    plt.xticks(rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",        "78%")
    col2.metric("Weighted F1",     "0.78")
    col3.metric("Cross-val F1",    "0.746 ± 0.096")
    col4.metric("High Risk Recall","100%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance")
        if os.path.exists('outputs/feature_importance.png'):
            st.image('outputs/feature_importance.png')
        else:
            st.warning("Run model.py first.")
    with col2:
        st.subheader("Confusion Matrix")
        if os.path.exists('outputs/confusion_matrix.png'):
            st.image('outputs/confusion_matrix.png')
        else:
            st.warning("Run model.py first.")

    st.markdown("---")
    st.subheader("SHAP Global Summary")
    if os.path.exists('outputs/shap_summary.png'):
        st.image('outputs/shap_summary.png')
    else:
        st.warning("Run model.py first.")

    st.markdown("---")
    st.subheader("Why SMOTE?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before SMOTE**")
        st.error("High Risk: 5 patients")
        st.warning("Medium Risk: 15 patients")
        st.success("Low Risk: 30 patients")
        st.markdown("High Risk Recall: **0%**")
    with col2:
        st.markdown("**After SMOTE**")
        st.error("High Risk: 30 patients (synthetic)")
        st.warning("Medium Risk: 30 patients (synthetic)")
        st.success("Low Risk: 30 patients")
        st.markdown("High Risk Recall: **100%**")

    st.info("SMOTE (Synthetic Minority Oversampling Technique) generates synthetic "
            "patient examples by interpolating between existing minority class samples. "
            "This ensures the model learns to identify High-risk patients rather than "
            "ignoring them due to their small number.")

    st.markdown("---")
    st.subheader("Model Pipeline")
    st.code("""
Raw CSVs: demographics + vitals_time_series + disease_risk_labels
    ↓
Merge on patient_id
    ↓
Aggregate vitals → 13 features (mean, max, min, std per vital)
    ↓
Compute trend features (heart_rate_trend, systolic_bp_trend, spo2_trend)
    ↓
Engineer clinical features (pulse_pressure, mean_arterial_pressure,
shock_index, spo2_danger_flag, instability_score)
    ↓
Encode categorical features (gender, smoking, diabetes, hypertension)
    ↓
SMOTE — balance classes to 30 each
    ↓
Train/Test Split — 80/20 stratified
    ↓
Random Forest Classifier — 300 trees, max_depth=10
    ↓
Evaluation: Accuracy 78%, Weighted F1 0.78, High Recall 100%
    ↓
SHAP TreeExplainer — per-patient waterfall explanations
    ↓
Streamlit Dashboard — 5 pages: Dashboard, Risk Prediction,
Patient Lookup, Data Analytics, Model Performance
    """, language="text")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("HealthGuard AI — IEEE DataPort Hackathon 2026 | "
           "For clinical decision support only. "
           "Not a substitute for medical diagnosis.")