"""
Clinical Readmission Risk + Hospital Resource Planning
Streamlit app — loads the trained pipeline (clinical_readmission_model.joblib)
and the CSVs produced by Clinical_readmission_risk.ipynb.

Run with:  streamlit run app.py
"""

import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap

st.set_page_config(page_title="Readmission Risk & Resource Planning", layout="wide")

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------
# Palette: warm-neutral paper background, deep clinical teal accent, muted
# status colors for risk bands. Typography: a serif for report-style headers
# (distinct from the generic SaaS sans-everywhere look), Inter for UI text,
# and IBM Plex Mono for every numeric/data value — a deliberate signature
# that reads as "measurement," consistent with how lab results are typeset.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #F4F5F1;
    --surface: #FFFFFF;
    --border: #DBDFD7;
    --text: #1B211D;
    --text-muted: #5B655E;
    --accent: #0F6E5C;
    --accent-dark: #0B5548;
    --low: #2F7D4F;
    --medium: #A9711F;
    --high: #A6392E;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp { background-color: var(--bg); }

h1, h2, h3 { font-family: 'Source Serif 4', serif; font-weight: 600; letter-spacing: -0.01em; }
h1 { border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; margin-bottom: 1.25rem; }

p, span, div, label { font-family: 'Inter', sans-serif; }

/* numeric values -- metrics, code, dataframes -- use the mono face */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], code, .stDataFrame, .stNumberInput input {
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif; color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }

/* tabs: flat underline nav instead of the default pill style */
.stTabs [data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif; font-weight: 500; color: var(--text-muted);
    background: transparent; padding: 0.5rem 0.1rem;
}
.stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom: 2px solid var(--accent) !important; }

/* buttons: flat, no gradient */
.stButton button, .stFormSubmitButton button, .stDownloadButton button {
    background: var(--accent); color: #FFFFFF; border: none; border-radius: 4px;
    font-family: 'Inter', sans-serif; font-weight: 500; padding: 0.55rem 1.1rem;
    box-shadow: none;
}
.stButton button:hover, .stFormSubmitButton button:hover, .stDownloadButton button:hover { background: var(--accent-dark); }

/* form + expander containers read as cards */
[data-testid="stForm"], .streamlit-expanderHeader {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
}

/* alerts: muted, no drop shadow, aligned with palette */
[data-testid="stAlert"] { border-radius: 4px; border: 1px solid var(--border); }

hr { border-color: var(--border); }

/* risk flag card -- the signature element, used for single-patient and cohort summaries */
.risk-flag { border: 1px solid var(--border); border-left: 4px solid var(--flag-color, var(--accent));
             border-radius: 4px; background: var(--surface); padding: 1rem 1.25rem; }
.risk-flag .band-label { font-family: 'Inter', sans-serif; font-size: 0.78rem; text-transform: uppercase;
                          letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 0.15rem; }
.risk-flag .band-value { font-family: 'Source Serif 4', serif; font-weight: 600; font-size: 1.4rem; color: var(--text); }
.risk-flag .proba-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; color: var(--text); }
</style>
""", unsafe_allow_html=True)


def render_risk_flag(band, proba):
    color_map = {"Low": "var(--low)", "Medium": "var(--medium)", "High": "var(--high)"}
    st.markdown(f"""
    <div class="risk-flag" style="--flag-color:{color_map[band]}">
        <div class="band-label">Predicted 30-day readmission probability</div>
        <div class="proba-value">{proba*100:.1f}%</div>
        <div class="band-label" style="margin-top:0.6rem;">Risk band</div>
        <div class="band-value" style="color:{color_map[band]}">{band}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
# NOTE: we deliberately do NOT load clinical_readmission_model.joblib here.
# That file pickles the XGBoost model's internal state, which is tied to the
# exact xgboost version that wrote it -- loading it with a different xgboost
# version (e.g. training vs. a different machine running this app) raises
# "XGBoostError: input stream corrupted". Instead we load the preprocessor
# (plain scikit-learn, pickles fine) and the classifier via XGBoost's own
# save_model()/load_model(), which IS designed to be version-portable.

@st.cache_resource
def load_model():
    preprocessor = joblib.load("preprocessor.joblib")
    with open("model_metadata.json") as f:
        meta = json.load(f)

    if meta["classifier_format"] == "xgboost_json":
        import xgboost as xgb
        clf = xgb.XGBClassifier()
        clf.load_model("classifier_model.json")
    else:
        clf = joblib.load("classifier_model.joblib")

    # ── sklearn 1.9 compatibility patch ────────────────────────────────────────
    # sklearn 1.9 calls np.isnan(ohe.categories_[i]) on every transform, which
    # crashes on object-dtype arrays that mix strings with float NaN. Fix: replace
    # float NaN in every category array with a sentinel string.
    _NAN_SENTINEL = "__MISSING__"
    _ohe = preprocessor.named_transformers_["cat"]
    for _i, _cats in enumerate(_ohe.categories_):
        if _cats.dtype == object and any(isinstance(v, float) and np.isnan(v) for v in _cats):
            _ohe.categories_[_i] = np.array(
                [_NAN_SENTINEL if (isinstance(v, float) and np.isnan(v)) else v for v in _cats],
                dtype=object,
            )
    # ───────────────────────────────────────────────────────────────────────────

    return preprocessor, clf, _NAN_SENTINEL


def _get_nan_sentinel_cols(preprocessor, nan_sentinel):
    """Return the set of categorical column names that use the sentinel."""
    ohe = preprocessor.named_transformers_["cat"]
    cat_cols = list(preprocessor.transformers_[1][2])
    return {col for i, col in enumerate(cat_cols) if nan_sentinel in ohe.categories_[i]}


def normalize_input_df(df: pd.DataFrame, preprocessor, nan_sentinel="__MISSING__", nan_sentinel_cols=None) -> pd.DataFrame:
    df = df.copy()
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])
    if nan_sentinel_cols is None:
        nan_sentinel_cols = _get_nan_sentinel_cols(preprocessor, nan_sentinel)

    for col in cat_cols:
        if col in df.columns:
            uses_sentinel = col in nan_sentinel_cols

            def _clean(val, _uses_sentinel=uses_sentinel):
                try:
                    is_missing = (val is None
                                  or (isinstance(val, float) and np.isnan(val))
                                  or str(val).strip() in ("", "nan", "NaN", "None",
                                                          "NULL", "null", "<NA>", "NA"))
                except Exception:
                    is_missing = False

                if is_missing:
                    return nan_sentinel if _uses_sentinel else "Missing"

                if isinstance(val, (int, float, np.integer, np.floating)):
                    if float(val).is_integer():
                        return str(int(val))
                    return str(val)

                return str(val).strip()

            df[col] = df[col].apply(_clean).astype(str)

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


class LoadedPipeline:
    """Thin wrapper so the rest of the app can keep calling model.predict_proba(X)."""
    def __init__(self, preprocessor, clf, nan_sentinel="__MISSING__"):
        self.preprocessor = preprocessor
        self.clf = clf
        self.nan_sentinel = nan_sentinel
        self.nan_sentinel_cols = _get_nan_sentinel_cols(preprocessor, nan_sentinel)

    def predict_proba(self, X):
        normalized = normalize_input_df(X, self.preprocessor,
                                        nan_sentinel=self.nan_sentinel,
                                        nan_sentinel_cols=self.nan_sentinel_cols)
        raw_cols = list(self.preprocessor.transformers_[0][2]) + list(self.preprocessor.transformers_[1][2])
        return self.clf.predict_proba(self.preprocessor.transform(normalized[raw_cols]))


@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


model = LoadedPipeline(*load_model())
test_risk_scores = load_csv("test_risk_scores.csv")
hospital_resource_plan = load_csv("hospital_resource_plan.csv")
department_resource_plan = load_csv("department_resource_plan.csv")


@st.cache_resource
def load_explainer(_clf):
    # TreeExplainer works directly on the fitted XGBoost classifier
    return shap.TreeExplainer(_clf)


explainer = load_explainer(model.clf)


def get_transformed_feature_names():
    num_cols = model.preprocessor.transformers_[0][2]
    cat_cols = model.preprocessor.transformers_[1][2]
    cat_names = list(model.preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    return list(num_cols) + cat_names


def get_raw_input_columns():
    """The original (pre-one-hot) column names the preprocessor expects as input."""
    return list(model.preprocessor.transformers_[0][2]) + list(model.preprocessor.transformers_[1][2])


def prettify_feature_name(name):
    for cat in model.preprocessor.transformers_[1][2]:
        prefix = cat + "_"
        if name.startswith(prefix):
            return f"{cat.replace('_', ' ')}: {name[len(prefix):]}"
    return name.replace("_", " ")


def top_key_factors(X_input, n=6):
    """Return the top-n factors pushing this patient's prediction up (risk-raising) and down (risk-lowering)."""
    X_t = model.preprocessor.transform(X_input)
    feature_names = get_transformed_feature_names()
    sv = explainer.shap_values(X_t)
    sv = sv[0] if sv.ndim > 1 else sv
    contrib = pd.DataFrame({"feature": feature_names, "shap_value": sv}).sort_values("shap_value", ascending=False)
    contrib = contrib[~contrib["feature"].str.startswith("gender_")]
    raising = contrib[contrib["shap_value"] > 0].head(n)
    lowering = contrib[contrib["shap_value"] < 0].tail(n).iloc[::-1]
    return raising, lowering

MEDICAL_SPECIALTIES = ['InternalMedicine', 'Family/GeneralPractice', 'Emergency/Trauma',
                        'Cardiology', 'Surgery-General', 'Orthopedics',
                        'Orthopedics-Reconstructive', 'Radiologist', 'Nephrology',
                        'Pulmonology', 'Other', 'Missing']

ADMISSION_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
                   5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped"}

DISCHARGE_DISPOSITION = {
    1: "Discharged to home", 2: "Transferred to another short term hospital",
    3: "Transferred to SNF", 4: "Transferred to ICF",
    5: "Transferred to another type of inpatient care institution",
    6: "Transferred to home with home health service", 7: "Left AMA",
    8: "Transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    12: "Still patient or expected to return for outpatient services",
    15: "Transferred within institution to Medicare approved swing bed",
    16: "Transferred/referred another institution for outpatient services",
    17: "Transferred/referred to this institution for outpatient services",
    18: "NULL", 22: "Transferred to another rehab facility",
    23: "Transferred to a long term care hospital",
    24: "Transferred to a nursing facility (Medicaid only)",
    25: "Not Mapped", 26: "Unknown/Invalid",
}

ADMISSION_SOURCE = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critical access hospital", 17: "NULL",
    20: "Not Mapped", 21: "Unknown/Invalid",
}

AGE_BANDS = [f'[{i}-{i+10})' for i in range(0, 100, 10)]
AGE_MAP = {b: i * 10 + 5 for i, b in enumerate(AGE_BANDS)}

HIGH_RISK_CATEGORIES = {'Circulatory', 'Diabetes', 'Respiratory', 'Genitourinary', 'Neoplasms'}


def icd9_category(code):
    """Same clinical-category mapping used to train the model."""
    if code is None or str(code).strip() == "":
        return 'Missing'
    code = str(code)
    if code.startswith('V') or code.startswith('E'):
        return 'External/Supplemental'
    try:
        val = float(code)
    except ValueError:
        return 'Other'
    if 390 <= val <= 459 or val == 785:
        return 'Circulatory'
    if 460 <= val <= 519 or val == 786:
        return 'Respiratory'
    if 520 <= val <= 579 or val == 787:
        return 'Digestive'
    if 250 <= val < 251:
        return 'Diabetes'
    if 800 <= val <= 999:
        return 'Injury'
    if 710 <= val <= 739:
        return 'Musculoskeletal'
    if 580 <= val <= 629 or val == 788:
        return 'Genitourinary'
    if 140 <= val <= 239:
        return 'Neoplasms'
    if 240 <= val <= 279:
        return 'Endocrine/Metabolic (other)'
    if 290 <= val <= 319:
        return 'Mental'
    if 320 <= val <= 389:
        return 'Nervous system'
    return 'Other'


def risk_band(p):
    if p < 0.33:
        return 'Low'
    if p < 0.66:
        return 'Medium'
    return 'High'


INTERVENTION_POLICY = {
    'High':   {'intervention': 'Case-manager led transition-of-care + home health referral',
               'follow_up_within_days': 2, 'staff_role': 'Case manager'},
    'Medium': {'intervention': 'Structured follow-up phone call + pharmacist medication review',
               'follow_up_within_days': 7, 'staff_role': 'Follow-up nurse'},
    'Low':    {'intervention': 'Standard discharge instructions',
               'follow_up_within_days': 30, 'staff_role': 'None (standard discharge)'},
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Clinical Readmission Risk & Hospital Resource Planning")

tab2, tab3, tab4 = st.tabs(["Patient Risk Predictor",
                             "Hospital Resource Plan", "Batch / Cohort Prioritization"])

# ---- Tab 2: Patient predictor ----------------------------------------------
with tab2:
    with st.expander("Key Factors Driving Readmission Risk (across all patients)"):
        st.caption("Global feature importance from the trained model — the factors clinicians should generally "
                   "pay closest attention to, distinct from the per-patient breakdown below.")
        try:
            feature_names = get_transformed_feature_names()
            importances = model.clf.feature_importances_
            imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
                      .sort_values("importance", ascending=False).head(15))
            imp_df["feature"] = imp_df["feature"].apply(prettify_feature_name)
            st.bar_chart(imp_df.set_index("feature")["importance"])
        except Exception as e:
            st.info(f"Couldn't compute global key factors ({e}).")

    st.subheader("Score a single patient")
    st.caption("Fields mirror exactly what the model was trained on (race is intentionally excluded).")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"])
            age_band = st.selectbox("Age band", AGE_BANDS, index=6)
            medical_specialty = st.selectbox("Medical specialty", MEDICAL_SPECIALTIES)
            admission_type_id = st.selectbox(
                "Admission type", list(ADMISSION_TYPE.keys()),
                format_func=lambda k: f"{k} — {ADMISSION_TYPE[k]}")
            discharge_disposition_id = st.selectbox(
                "Discharge disposition", list(DISCHARGE_DISPOSITION.keys()),
                format_func=lambda k: f"{k} — {DISCHARGE_DISPOSITION[k]}")
            admission_source_id = st.selectbox(
                "Admission source", list(ADMISSION_SOURCE.keys()),
                format_func=lambda k: f"{k} — {ADMISSION_SOURCE[k]}")

        with c2:
            time_in_hospital = st.number_input("Time in hospital (days)", 1, 14, 4)
            num_lab_procedures = st.number_input("Number of lab procedures", 0, 150, 45)
            num_procedures = st.number_input("Number of procedures", 0, 10, 1)
            num_medications = st.number_input("Number of medications", 0, 100, 15)
            number_outpatient = st.number_input("Outpatient visits (prior year)", 0, 50, 0)
            number_emergency = st.number_input("Emergency visits (prior year)", 0, 50, 0)
            number_inpatient = st.number_input("Inpatient visits (prior year)", 0, 50, 0)
            number_diagnoses = st.number_input("Number of diagnoses", 1, 20, 7)

        with c3:
            max_glu_serum = st.selectbox("Max glucose serum", ["None", "Norm", ">200", ">300"])
            A1Cresult = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
            change = st.selectbox("Medication changed at this visit", ["No", "Ch"])
            diabetesMed = st.selectbox("On diabetes medication", ["No", "Yes"])
            num_meds_active = st.number_input("Number of active diabetes medications", 0, 23, 1)
            num_meds_changed = st.number_input("Number of medications changed (Up/Down)", 0, 23, 0)

        st.markdown("**Diagnosis codes (ICD-9, optional — leave blank if unknown)**")
        d1, d2, d3 = st.columns(3)
        diag_1 = d1.text_input("diag_1 (primary)", "428")
        diag_2 = d2.text_input("diag_2", "250")
        diag_3 = d3.text_input("diag_3", "")

        submitted = st.form_submit_button("Predict readmission risk", type="primary")

    if submitted:
        diag_1_cat = icd9_category(diag_1)
        diag_2_cat = icd9_category(diag_2)
        diag_3_cat = icd9_category(diag_3)
        cats = [diag_1_cat, diag_2_cat, diag_3_cat]
        n_high_risk_dx = sum(c in HIGH_RISK_CATEGORIES for c in cats)

        row = {
            'gender': gender,
            'admission_type_id': str(admission_type_id),
            'discharge_disposition_id': str(discharge_disposition_id),
            'admission_source_id': str(admission_source_id),
            'medical_specialty': medical_specialty,
            'max_glu_serum': max_glu_serum,
            'A1Cresult': A1Cresult,
            'change': change,
            'diabetesMed': diabetesMed,
            'diag_1_cat': diag_1_cat,
            'diag_2_cat': diag_2_cat,
            'diag_3_cat': diag_3_cat,
            'age_mid': AGE_MAP[age_band],
            'time_in_hospital': time_in_hospital,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_outpatient': number_outpatient,
            'number_emergency': number_emergency,
            'number_inpatient': number_inpatient,
            'number_diagnoses': number_diagnoses,
            'n_high_risk_dx': n_high_risk_dx,
            'has_circulatory_dx': int('Circulatory' in cats),
            'has_diabetes_dx': int('Diabetes' in cats),
            'has_respiratory_dx': int('Respiratory' in cats),
            'has_genitourinary_dx': int('Genitourinary' in cats),
            'has_neoplasm_dx': int('Neoplasms' in cats),
            'high_risk_disease_flag': int(n_high_risk_dx > 0),
            'num_meds_active': num_meds_active,
            'num_meds_changed': num_meds_changed,
        }
        X_input = pd.DataFrame([row])

        proba = model.predict_proba(X_input)[0, 1]
        band = risk_band(proba)
        policy = INTERVENTION_POLICY[band]

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            render_risk_flag(band, proba)
        with r2:
            st.write("**Recommended intervention (from the resource-planning policy):**")
            st.write(f"- {policy['intervention']}")
            st.write(f"- Follow up within **{policy['follow_up_within_days']} days**")
            st.write(f"- Assign: **{policy['staff_role']}**")

        st.subheader("Key Factors for This Patient")
        st.caption("What's pushing this specific patient's risk up or down — use this to target the intervention "
                   "instead of applying the same follow-up plan to everyone in a risk band.")
        raising, lowering = top_key_factors(X_input, n=6)
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**Increasing risk**")
            if len(raising) > 0:
                for _, r in raising.iterrows():
                    st.write(f"- {prettify_feature_name(r['feature'])}")
            else:
                st.write("_No strong risk-raising factors identified._")
        with f2:
            st.markdown("**Lowering risk**")
            if len(lowering) > 0:
                for _, r in lowering.iterrows():
                    st.write(f"- {prettify_feature_name(r['feature'])}")
            else:
                st.write("_No strong risk-lowering factors identified._")

# ---- Tab 3: Resource plan ---------------------------------------------------
def compute_cohort_resource_forecast(results):
    if "time_in_hospital" in results.columns:
        los = pd.to_numeric(results["time_in_hospital"], errors="coerce").fillna(4.0)
    else:
        los = pd.Series(4.0, index=results.index)

    probs = results["predicted_probability"]
    by_band = {}
    bands = ["High", "Medium", "Low"]

    for b in bands:
        sub = results[results["risk_band"] == b]
        n_pts = len(sub)
        if n_pts > 0:
            sub_probs = sub["predicted_probability"]
            sub_los = los.loc[sub.index]
            exp_readmit = float(sub_probs.sum())
            exp_bed_days = float((sub_probs * sub_los).sum())
        else:
            exp_readmit = 0.0
            exp_bed_days = 0.0

        if b == "High":
            staff_req = math.ceil(n_pts / 20.0) if n_pts > 0 else 0
            workload = "Case-manager led transition-of-care + home health referral (within 2 days)"
        elif b == "Medium":
            staff_req = math.ceil(n_pts / 60.0) if n_pts > 0 else 0
            workload = "Structured follow-up phone call + pharmacist medication review (within 7 days)"
        else:
            staff_req = 0
            workload = "Standard discharge instructions (within 30 days)"

        by_band[b] = {
            "n_patients": n_pts,
            "expected_readmissions": round(exp_readmit, 2),
            "estimated_bed_days": round(exp_bed_days, 1),
            "staff_required": staff_req,
            "workload": workload,
        }

    total_pts = len(results)
    total_exp_readmit = float(probs.sum()) if total_pts > 0 else 0.0
    total_exp_bed_days = float((probs * los).sum()) if total_pts > 0 else 0.0

    return {
        "total_patients": total_pts,
        "total_expected_readmissions": round(total_exp_readmit, 2),
        "total_estimated_bed_days": round(total_exp_bed_days, 1),
        "by_band": by_band,
    }

def render_capacity_vs_demand(forecast):
    if not forecast:
        return
    st.subheader("Hospital Capacity vs Demand")
    st.caption("Operational comparison based on predicted ML cohort demand and user-configured capacity.")

    with st.expander("Configure Hospital Capacity Inputs", expanded=False):
        c1, c2, c3 = st.columns(3)
        total_beds = c1.number_input("Total Hospital Beds", value=200, min_value=1)
        occupied_beds = c2.number_input("Currently Occupied Beds", value=150, min_value=0)
        available_beds = max(0, total_beds - occupied_beds)
        c3.number_input("Available Beds (Calculated)", value=available_beds, disabled=True)

        c4, c5, c6 = st.columns(3)
        c4.number_input("Total ICU Beds", value=30, min_value=0)
        c5.number_input("Available ICU Beds", value=10, min_value=0)
        nurses_avail = c6.number_input("Nurses Available", value=20, min_value=0)

        c7, c8, c9 = st.columns(3)
        c7.number_input("Doctors Available", value=10, min_value=0)
        cm_avail = c8.number_input("Case Managers Available", value=2, min_value=0)
        c9.number_input("Pharmacists Available", value=3, min_value=0)

    bed_demand = forecast.get("total_estimated_bed_days", 0.0)
    if available_beds < bed_demand:
        bed_status = "SHORTAGE"
    elif available_beds <= bed_demand * 1.15:
        bed_status = "NEAR CAPACITY"
    else:
        bed_status = "SUFFICIENT"
    add_beds = math.ceil(bed_demand - available_beds) if available_beds < bed_demand else 0

    cm_req = forecast.get("by_band", {}).get("High", {}).get("staff_required", 0)
    cm_status = "SHORTAGE" if cm_avail < cm_req else ("NEAR CAPACITY" if cm_avail == cm_req else "SUFFICIENT")
    cm_add = max(0, cm_req - cm_avail)

    nurse_req = forecast.get("by_band", {}).get("Medium", {}).get("staff_required", 0)
    nurse_status = "SHORTAGE" if nurses_avail < nurse_req else ("NEAR CAPACITY" if nurses_avail == nurse_req else "SUFFICIENT")
    nurse_add = max(0, nurse_req - nurses_avail)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Bed Capacity ({bed_status})")
        st.write(f"- Available beds: **{available_beds}**")
        st.write(f"- Estimated demand: **{bed_demand:.1f} bed-days**")
        if add_beds > 0:
            st.error(f"Additional beds required: {add_beds}")
        else:
            st.success(f"Surplus capacity: {available_beds - bed_demand:.1f} beds")

    with col2:
        st.markdown(f"### Case Managers ({cm_status})")
        st.write(f"- Required: **{cm_req}**")
        st.write(f"- Available: **{cm_avail}**")
        if cm_add > 0:
            st.error(f"Additional required: {cm_add}")
        else:
            st.success(f"Surplus staff: {cm_avail - cm_req}")

    with col3:
        st.markdown(f"### Follow-up Staff ({nurse_status})")
        st.write(f"- Required: **{nurse_req}**")
        st.write(f"- Available: **{nurses_avail}**")
        if nurse_add > 0:
            st.error(f"Additional required: {nurse_add}")
        else:
            st.success(f"Surplus staff: {nurses_avail - nurse_req}")

# ---- Tab 3: Resource plan ---------------------------------------------------
with tab3:
    st.subheader("Hospital Resource Plan")
    active_plan = st.session_state.get("active_cohort_resource_forecast")
    if active_plan:
        st.info(f"Displaying resource forecast dynamically derived from the current uploaded patient cohort ({active_plan['total_patients']} patients).")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Patients", active_plan["total_patients"])
        m2.metric("High Risk", active_plan["by_band"]["High"]["n_patients"])
        m3.metric("Medium Risk", active_plan["by_band"]["Medium"]["n_patients"])
        m4.metric("Low Risk", active_plan["by_band"]["Low"]["n_patients"])
        m5.metric("Expected Readmissions", active_plan["total_expected_readmissions"])
        m6.metric("Estimated Bed-Days", active_plan["total_estimated_bed_days"])

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown("### High Risk Resources")
            st.write(f"- Patients: **{active_plan['by_band']['High']['n_patients']}**")
            st.write(f"- Expected readmissions: **{active_plan['by_band']['High']['expected_readmissions']}**")
            st.write(f"- Estimated bed-days: **{active_plan['by_band']['High']['estimated_bed_days']}**")
            st.write(f"- Case managers required: **{active_plan['by_band']['High']['staff_required']}**")
            st.caption(f"Workload: {active_plan['by_band']['High']['workload']}")
        with r2:
            st.markdown("### Medium Risk Resources")
            st.write(f"- Patients: **{active_plan['by_band']['Medium']['n_patients']}**")
            st.write(f"- Expected readmissions: **{active_plan['by_band']['Medium']['expected_readmissions']}**")
            st.write(f"- Estimated bed-days: **{active_plan['by_band']['Medium']['estimated_bed_days']}**")
            st.write(f"- Follow-up staff required: **{active_plan['by_band']['Medium']['staff_required']}**")
            st.caption(f"Workload: {active_plan['by_band']['Medium']['workload']}")
        with r3:
            st.markdown("### Low Risk Resources")
            st.write(f"- Patients: **{active_plan['by_band']['Low']['n_patients']}**")
            st.write(f"- Expected readmissions: **{active_plan['by_band']['Low']['expected_readmissions']}**")
            st.write(f"- Estimated bed-days: **{active_plan['by_band']['Low']['estimated_bed_days']}**")
            st.caption(f"Standard follow-up: {active_plan['by_band']['Low']['workload']}")
        st.divider()

    render_capacity_vs_demand(active_plan if active_plan else {"total_estimated_bed_days": 186.6, "by_band": {"High": {"staff_required": 2}, "Medium": {"staff_required": 4}}})

    st.subheader("Static Hospital Reference Plan")
    if hospital_resource_plan is not None:
        st.dataframe(hospital_resource_plan, use_container_width=True)
        if 'recommended_fte' in hospital_resource_plan.columns:
            st.bar_chart(hospital_resource_plan.set_index(hospital_resource_plan.columns[0])['recommended_fte'])
    else:
        st.info("hospital_resource_plan.csv not found — run the notebook first to generate it.")

    st.subheader("Department-level allocation")
    if department_resource_plan is not None:
        st.dataframe(department_resource_plan, use_container_width=True)
        if {'medical_specialty', 'expected_bed_days'}.issubset(department_resource_plan.columns):
            top = (department_resource_plan.groupby('medical_specialty')['expected_bed_days']
                   .sum().sort_values(ascending=False).head(10))
            st.write("**Projected Department Bed Demand**")
            st.bar_chart(top)
    else:
        st.info("department_resource_plan.csv not found — run the notebook first to generate it.")


# ---- Tab 4: Batch / cohort prioritization -----------------------------------
with tab4:
    st.subheader("Score a Whole Patient Cohort & Forecast Resources")
    st.caption(
        "Upload a CSV of multiple patients (e.g. today's discharge list) to score them all at once "
        "and automatically generate hospital resource planning forecasts based on the actual predictions."
    )

    required_cols = get_raw_input_columns()
    with st.expander("Required CSV columns"):
        st.write(", ".join(required_cols))

    uploaded = st.file_uploader("Upload patient cohort CSV", type=["csv"])

    if uploaded is not None:
        try:
            cohort_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            cohort_df = None

        if cohort_df is not None:
            missing_cols = [c for c in required_cols if c not in cohort_df.columns]
            if missing_cols:
                st.error(f"Uploaded file is missing {len(missing_cols)} required column(s): "
                          + ", ".join(missing_cols[:10]) + (" ..." if len(missing_cols) > 10 else ""))
            else:
                with st.spinner("Scoring cohort and generating resource plan..."):
                    X_batch = cohort_df[required_cols].copy()
                    probas = model.predict_proba(X_batch)[:, 1]
                    results = cohort_df.copy()
                    results["predicted_probability"] = probas
                    results["risk_band"] = [risk_band(p) for p in probas]
                    results["recommended_intervention"] = [INTERVENTION_POLICY[b]["intervention"] for b in results["risk_band"]]
                    results["follow_up_within_days"] = [INTERVENTION_POLICY[b]["follow_up_within_days"] for b in results["risk_band"]]
                    forecast = compute_cohort_resource_forecast(results)
                    st.session_state["active_cohort_resource_forecast"] = forecast

                st.success(f"Scored {len(results)} patients and generated resource forecast.")

                st.subheader("Cohort Resource Forecast Dashboard")
                st.markdown(f"**COHORT: {forecast['total_patients']} PATIENTS**")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total Patients", forecast["total_patients"])
                m2.metric("High Risk", forecast["by_band"]["High"]["n_patients"])
                m3.metric("Medium Risk", forecast["by_band"]["Medium"]["n_patients"])
                m4.metric("Low Risk", forecast["by_band"]["Low"]["n_patients"])
                m5.metric("Expected Readmissions", forecast["total_expected_readmissions"])
                m6.metric("Estimated Bed-Days", forecast["total_estimated_bed_days"])

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown("#### High Risk Resources")
                    st.write(f"- Expected readmissions: **{forecast['by_band']['High']['expected_readmissions']}**")
                    st.write(f"- Expected bed-days (est): **{forecast['by_band']['High']['estimated_bed_days']}**")
                    st.write(f"- Case managers required: **{forecast['by_band']['High']['staff_required']}**")
                    st.caption(f"Workload: {forecast['by_band']['High']['workload']}")
                with r2:
                    st.markdown("#### Medium Risk Resources")
                    st.write(f"- Expected readmissions: **{forecast['by_band']['Medium']['expected_readmissions']}**")
                    st.write(f"- Expected bed-days (est): **{forecast['by_band']['Medium']['estimated_bed_days']}**")
                    st.write(f"- Follow-up staff required: **{forecast['by_band']['Medium']['staff_required']}**")
                    st.caption(f"Workload: {forecast['by_band']['Medium']['workload']}")
                with r3:
                    st.markdown("#### Low Risk Resources")
                    st.write(f"- Expected readmissions: **{forecast['by_band']['Low']['expected_readmissions']}**")
                    st.write(f"- Expected bed-days (est): **{forecast['by_band']['Low']['estimated_bed_days']}**")
                    st.caption(f"Standard follow-up: {forecast['by_band']['Low']['workload']}")

                st.subheader("Prioritized Patient List (highest risk first)")
                id_cols = [c for c in ["encounter_id", "patient_nbr"] if c in results.columns]
                display_cols = id_cols + ["predicted_probability", "risk_band", "recommended_intervention", "follow_up_within_days"]
                prioritized = results.sort_values("predicted_probability", ascending=False)[display_cols]
                st.dataframe(prioritized, use_container_width=True, height=400)

                csv_bytes = prioritized.to_csv(index=False).encode("utf-8")
                st.download_button("Download prioritized risk list (CSV)", csv_bytes,
                                    file_name="prioritized_readmission_risk_list.csv", mime="text/csv",
                                    use_container_width=True)
    else:
        st.info("No file uploaded yet. Once you upload a cohort CSV, you'll see a prioritized, "
                "downloadable risk list plus dynamic resource planning forecasts.")

