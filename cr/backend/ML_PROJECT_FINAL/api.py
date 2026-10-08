import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io

BASE = Path(__file__).resolve().parent
latest_cohort_resource_plan = None

app = FastAPI(title="Clinical Readmission Risk API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = joblib.load(BASE / "preprocessor.joblib")
with open(BASE / "model_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

if metadata.get("classifier_format") == "xgboost_json":
    classifier = xgb.XGBClassifier()
    classifier.load_model(str(BASE / "classifier_model.json"))
else:
    classifier = joblib.load(BASE / "classifier_model.joblib")

explainer = shap.TreeExplainer(classifier)


_NAN_SENTINEL = "__MISSING__"
_ohe = preprocessor.named_transformers_["cat"]
for _i, _cats in enumerate(_ohe.categories_):
    if _cats.dtype == object and any(isinstance(v, float) and np.isnan(v) for v in _cats):
        _ohe.categories_[_i] = np.array(
            [_NAN_SENTINEL if (isinstance(v, float) and np.isnan(v)) else v for v in _cats],
            dtype=object,
        )
_NAN_SENTINEL_COLS = set()
_cat_cols_all = list(preprocessor.transformers_[1][2])
for _i, _col in enumerate(_cat_cols_all):
    if _NAN_SENTINEL in _ohe.categories_[_i]:
        _NAN_SENTINEL_COLS.add(_col)

MEDICAL_SPECIALTIES = [
    "InternalMedicine", "Family/GeneralPractice", "Emergency/Trauma",
    "Cardiology", "Surgery-General", "Orthopedics",
    "Orthopedics-Reconstructive", "Radiologist", "Nephrology",
    "Pulmonology", "Other", "Missing"
]
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
AGE_BANDS = [f"[{i}-{i+10})" for i in range(0, 100, 10)]
AGE_MAP = {b: i * 10 + 5 for i, b in enumerate(AGE_BANDS)}
HIGH_RISK_CATEGORIES = {"Circulatory", "Diabetes", "Respiratory", "Genitourinary", "Neoplasms"}

INTERVENTION_POLICY = {
    "High": {"intervention": "Case-manager led transition-of-care + home health referral",
             "follow_up_within_days": 2, "staff_role": "Case manager", "concurrent_capacity_per_fte": 20},
    "Medium": {"intervention": "Structured follow-up phone call + pharmacist medication review",
                "follow_up_within_days": 7, "staff_role": "Follow-up nurse", "concurrent_capacity_per_fte": 60},
    "Low": {"intervention": "Standard discharge instructions",
            "follow_up_within_days": 30, "staff_role": "None (standard discharge)", "concurrent_capacity_per_fte": None},
}

# Default assumption: how many days the uploaded/loaded cohort is treated as spanning.
# The model predicts 30-day readmission risk, so 30 is the natural default -- but it's an
# explicit, adjustable assumption, not something the data tells us. Without a horizon, a
# cumulative "bed-days" total has no time unit and can't be compared to a snapshot beds-available
# count -- that mismatch was the root cause of the bed-shortage bug.
DEFAULT_HORIZON_DAYS = 30

RAW_COLUMNS = list(preprocessor.transformers_[0][2]) + list(preprocessor.transformers_[1][2])

def icd9_category(code):
    if code is None or str(code).strip() == "":
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "External/Supplemental"
    try:
        val = float(code)
    except ValueError:
        return "Other"
    if 390 <= val <= 459 or val == 785: return "Circulatory"
    if 460 <= val <= 519 or val == 786: return "Respiratory"
    if 520 <= val <= 579 or val == 787: return "Digestive"
    if 250 <= val < 251: return "Diabetes"
    if 800 <= val <= 999: return "Injury"
    if 710 <= val <= 739: return "Musculoskeletal"
    if 580 <= val <= 629 or val == 788: return "Genitourinary"
    if 140 <= val <= 239: return "Neoplasms"
    if 240 <= val <= 279: return "Endocrine/Metabolic (other)"
    if 290 <= val <= 319: return "Mental"
    if 320 <= val <= 389: return "Nervous system"
    return "Other"

def risk_band(p):
    if p < 0.33: return "Low"
    if p < 0.66: return "Medium"
    return "High"

def prettify(name):
    for cat in preprocessor.transformers_[1][2]:
        prefix = cat + "_"
        if name.startswith(prefix):
            return f"{cat.replace('_', ' ')}: {name[len(prefix):]}"
    return name.replace("_", " ")

def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw DataFrame so it is compatible with the loaded preprocessor.

    Key rules:
    - Categorical ID columns (admission_type_id etc.) are stored as strings in
      the trained OHE, so numeric values from pd.read_csv are cast to str.
    - max_glu_serum / A1Cresult: missing values map to _NAN_SENTINEL (the
      string that replaced the float-NaN category in ohe.categories_).
    - All other categorical columns: missing → 'Missing'.
    - Numeric columns: coerce to float, fill NaN with 0.
    """
    df = df.copy()
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])

    for col in cat_cols:
        if col in df.columns:
            uses_sentinel = col in _NAN_SENTINEL_COLS

            def _clean(val, _uses_sentinel=uses_sentinel, _col=col):
                # Detect every flavour of "missing"
                try:
                    is_missing = (val is None
                                  or (isinstance(val, float) and np.isnan(val))
                                  or str(val).strip() in ("", "nan", "NaN", "None",
                                                          "NULL", "null", "<NA>", "NA"))
                except Exception:
                    is_missing = False

                if is_missing:
                    return _NAN_SENTINEL if _uses_sentinel else "Missing"

                # Integer-like floats from pd.read_csv (e.g. 1.0 → '1')
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


def predict_df(df):
    normalized = normalize_input_df(df)
    probs = classifier.predict_proba(preprocessor.transform(normalized[RAW_COLUMNS]))[:, 1]
    return probs

def patient_from_ui(d):
    cats = [icd9_category(d.get("diag_1")), icd9_category(d.get("diag_2")), icd9_category(d.get("diag_3"))]
    n_high = sum(c in HIGH_RISK_CATEGORIES for c in cats)
    return {
        "gender": d.get("gender", "Unknown/Invalid"),
        "admission_type_id": str(d["admission_type_id"]),
        "discharge_disposition_id": str(d["discharge_disposition_id"]),
        "admission_source_id": str(d["admission_source_id"]),
        "medical_specialty": d.get("medical_specialty", "Missing"),
        "max_glu_serum": _NAN_SENTINEL if d.get("max_glu_serum") in (None, "None", "", "nan", "NaN") else str(d.get("max_glu_serum")).strip(),
        "A1Cresult": _NAN_SENTINEL if d.get("A1Cresult") in (None, "None", "", "nan", "NaN") else str(d.get("A1Cresult")).strip(),

        "change": d.get("change", "No"),
        "diabetesMed": d.get("diabetesMed", "No"),
        "diag_1_cat": cats[0], "diag_2_cat": cats[1], "diag_3_cat": cats[2],
        "age_mid": AGE_MAP.get(d.get("age_band"), 65),
        "time_in_hospital": d.get("time_in_hospital", 4),
        "num_lab_procedures": d.get("num_lab_procedures", 45),
        "num_procedures": d.get("num_procedures", 1),
        "num_medications": d.get("num_medications", 15),
        "number_outpatient": d.get("number_outpatient", 0),
        "number_emergency": d.get("number_emergency", 0),
        "number_inpatient": d.get("number_inpatient", 0),
        "number_diagnoses": d.get("number_diagnoses", 7),
        "n_high_risk_dx": n_high,
        "has_circulatory_dx": int("Circulatory" in cats),
        "has_diabetes_dx": int("Diabetes" in cats),
        "has_respiratory_dx": int("Respiratory" in cats),
        "has_genitourinary_dx": int("Genitourinary" in cats),
        "has_neoplasm_dx": int("Neoplasms" in cats),
        "high_risk_disease_flag": int(n_high > 0),
        "num_meds_active": d.get("num_meds_active", 1),
        "num_meds_changed": d.get("num_meds_changed", 0),
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "model": metadata.get("best_model", "XGBoost")}

@app.get("/api/options")
def options():
    return {
        "raw_columns": RAW_COLUMNS,
        "age_bands": AGE_BANDS,
        "medical_specialties": MEDICAL_SPECIALTIES,
        "admission_types": [{"id": k, "label": v} for k, v in ADMISSION_TYPE.items()],
        "discharge_dispositions": [{"id": k, "label": v} for k, v in DISCHARGE_DISPOSITION.items()],
        "admission_sources": [{"id": k, "label": v} for k, v in ADMISSION_SOURCE.items()],
        "glucose": ["None", "Norm", ">200", ">300"],
        "a1c": ["None", "Norm", ">7", ">8"],
    }

@app.post("/api/predict")
def predict(payload: dict[str, Any]):
    try:
        row = patient_from_ui(payload)
        X = pd.DataFrame([row])
        normalized = normalize_input_df(X)
        p = float(predict_df(normalized)[0])
        band = risk_band(p)
        policy = INTERVENTION_POLICY[band]

        Xt = preprocessor.transform(normalized[RAW_COLUMNS])
        names = list(preprocessor.transformers_[0][2]) + list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(preprocessor.transformers_[1][2])
        )
        sv = explainer.shap_values(Xt)
        sv = np.asarray(sv)
        if sv.ndim > 1: sv = sv[0]
        contributions = pd.DataFrame({"feature": names, "value": sv})
        contributions = contributions[~contributions.feature.str.startswith("gender_")]
        raising = contributions[contributions.value > 0].sort_values("value", ascending=False).head(6)
        lowering = contributions[contributions.value < 0].sort_values("value", ascending=True).head(6)

        return {
            "probability": round(p * 100, 1),
            "band": band,
            "intervention": policy["intervention"],
            "follow_up_within_days": policy["follow_up_within_days"],
            "staff_role": policy["staff_role"],
            "increasing_risk": [prettify(x) for x in raising.feature.tolist()],
            "lowering_risk": [prettify(x) for x in lowering.feature.tolist()],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def compute_resource_plan(results: pd.DataFrame, horizon_days: float = DEFAULT_HORIZON_DAYS) -> dict:
    """Turn per-patient predictions into a resource plan.

    `horizon_days` is the number of days this cohort is assumed to represent (e.g. "this
    is a week of discharges" -> 7). It's what makes the output comparable to a snapshot of
    beds/staff on hand:
    - bed-days (a cumulative total with no time unit) -> avg beds needed *per day*
    - total patient headcount -> *concurrent* caseload, via Little's Law (arrival rate x how
      long each patient stays in the follow-up program), which is what a staffing ratio like
      "20 patients per case manager" actually means.
    Without this conversion, a multi-thousand-patient cumulative total was being compared
    directly to a beds-available count and would report a shortage almost no matter what.
    """
    if horizon_days is None or horizon_days <= 0:
        horizon_days = DEFAULT_HORIZON_DAYS

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
        policy = INTERVENTION_POLICY[b]
        if n_pts > 0:
            sub_probs = sub["predicted_probability"]
            sub_los = los.loc[sub.index]
            exp_readmit = float(sub_probs.sum())
            exp_bed_days = float((sub_probs * sub_los).sum())
            avg_los = float(sub_los.mean())
        else:
            exp_readmit = 0.0
            exp_bed_days = 0.0
            avg_los = 0.0

        avg_daily_beds_needed = exp_bed_days / horizon_days
        daily_arrivals = n_pts / horizon_days
        concurrent_caseload = daily_arrivals * policy["follow_up_within_days"]
        capacity_per_fte = policy["concurrent_capacity_per_fte"]
        staff_req = math.ceil(concurrent_caseload / capacity_per_fte) if capacity_per_fte else 0

        by_band[b] = {
            "risk_band": b,
            "n_patients": n_pts,
            "expected_readmissions": round(exp_readmit, 2),
            "estimated_bed_days": round(exp_bed_days, 1),
            "avg_daily_beds_needed": round(avg_daily_beds_needed, 2),
            "avg_time_in_hospital": round(avg_los, 2),
            "concurrent_caseload": round(concurrent_caseload, 1),
            "staff_required": staff_req,
            "staff_label": f"Recommended {policy['staff_role'].lower()}s" if capacity_per_fte else "Standard follow-up",
            "staff_role": policy["staff_role"],
            "workload": f"{policy['intervention']} (within {policy['follow_up_within_days']} days)",
        }

    total_pts = len(results)
    total_exp_readmit = float(probs.sum()) if total_pts > 0 else 0.0
    total_exp_bed_days = float((probs * los).sum()) if total_pts > 0 else 0.0
    total_avg_daily_beds_needed = total_exp_bed_days / horizon_days

    avg_prob = float(probs.mean()) if total_pts > 0 else 0.0
    n_high_risk = by_band["High"]["n_patients"]
    n_medium_risk = by_band["Medium"]["n_patients"]
    n_low_risk = by_band["Low"]["n_patients"]

    high_risk_bed_days = by_band["High"]["estimated_bed_days"]
    icu_bed_days = high_risk_bed_days * 0.15
    avg_daily_icu_beds_needed = icu_bed_days / horizon_days
    est_icu_beds_required = math.ceil(avg_daily_icu_beds_needed)

    assumptions = [
        f"Planning horizon: Cohort spans {horizon_days:.0f} discharge days.",
        "Readmission Risk Bands: Low (p < 0.33), Medium (0.33 <= p < 0.66), High (p >= 0.66).",
        "Expected Readmissions: SUM(p_i) across all uploaded patients.",
        "Expected Bed-Days: SUM(p_i * time_in_hospital_i). Default LOS = 4.0 days if missing.",
        "Average Daily Bed Demand: Expected Bed-Days divided by planning horizon days.",
        "ICU Bed Requirement: Estimated prototype acuity assumption (~15% of high-risk bed-days require ICU monitoring).",
        "Case Manager Staffing (High Risk): Derived via Little's Law with 2-day follow-up window and 20 patient concurrent capacity per FTE.",
        "Follow-up Nurse Staffing (Medium Risk): Derived via Little's Law with 7-day follow-up window and 60 patient concurrent capacity per FTE."
    ]

    hospital_rows = []
    for b in ["Low", "Medium", "High"]:
        b_info = by_band[b]
        sub = results[results["risk_band"] == b]
        hospital_rows.append({
            "risk_category": b,
            "n_patients": b_info["n_patients"],
            "avg_time_in_hospital": b_info["avg_time_in_hospital"],
            "expected_readmissions": b_info["expected_readmissions"],
            "predicted_readmit_rate": round(float(sub["predicted_probability"].mean()), 2) if b_info["n_patients"] > 0 else 0.0,
            "expected_bed_days": b_info["estimated_bed_days"],
            "avg_daily_beds_needed": b_info["avg_daily_beds_needed"],
            "intervention": b_info["workload"],
            "follow_up_within_days": INTERVENTION_POLICY[b]["follow_up_within_days"],
            "staff_role": b_info["staff_role"],
            "concurrent_caseload": b_info["concurrent_caseload"],
            "recommended_fte": b_info["staff_required"],
        })

    dept_rows = []
    if "medical_specialty" in results.columns:
        dept_grp = results.groupby(["medical_specialty", "risk_band"], observed=True)
        for (dept, band), g in dept_grp:
            g_los = los.loc[g.index]
            g_probs = g["predicted_probability"]
            g_bed_days = float((g_probs * g_los).sum())
            dept_rows.append({
                "medical_specialty": str(dept),
                "risk_category": str(band),
                "n_patients": len(g),
                "avg_time_in_hospital": round(float(g_los.mean()), 2),
                "expected_readmissions": round(float(g_probs.sum()), 2),
                "predicted_readmit_rate": round(float(g_probs.mean()), 2),
                "expected_bed_days": round(g_bed_days, 1),
                "avg_daily_beds_needed": round(g_bed_days / horizon_days, 2),
            })
        dept_rows.sort(key=lambda x: x["expected_bed_days"], reverse=True)

    return {
        "is_cohort": True,
        "horizon_days": horizon_days,
        "planning_horizon": horizon_days,
        "total_patients": total_pts,
        "predicted_readmissions": round(total_exp_readmit, 2),
        "expected_readmissions": round(total_exp_readmit, 2),
        "total_expected_readmissions": round(total_exp_readmit, 2),
        "average_predicted_probability": round(avg_prob * 100, 1),
        "predicted_high_risk_patients": n_high_risk,
        "predicted_medium_risk_patients": n_medium_risk,
        "predicted_low_risk_patients": n_low_risk,
        "total_estimated_bed_days": round(total_exp_bed_days, 1),
        "total_avg_daily_beds_needed": round(total_avg_daily_beds_needed, 2),
        "beds_required": round(total_avg_daily_beds_needed, 2),
        "icu_bed_days": round(icu_bed_days, 1),
        "avg_daily_icu_beds_needed": round(avg_daily_icu_beds_needed, 2),
        "icu_beds_required": est_icu_beds_required,
        "staff_required": {
            "case_managers": by_band["High"]["staff_required"],
            "follow_up_nurses": by_band["Medium"]["staff_required"],
            "total_fte": by_band["High"]["staff_required"] + by_band["Medium"]["staff_required"]
        },
        "assumptions": assumptions,
        "by_band": by_band,
        "hospital": hospital_rows,
        "department": dept_rows,
    }

@app.get("/api/resource")
def resource():
    global latest_cohort_resource_plan
    hospital = pd.read_csv(BASE / "hospital_resource_plan.csv")
    department = pd.read_csv(BASE / "department_resource_plan.csv")

    static_by_band = {
        row["risk_category"]: {"staff_required": int(row.get("recommended_fte", 0) or 0)}
        for row in hospital.to_dict(orient="records")
    }
    static_plan = {
        "is_cohort": False,
        "horizon_days": DEFAULT_HORIZON_DAYS,
        "total_estimated_bed_days": round(float(hospital["expected_bed_days"].sum()), 1),
        "total_avg_daily_beds_needed": round(float(hospital.get("avg_daily_beds_needed", hospital["expected_bed_days"] / DEFAULT_HORIZON_DAYS).sum()), 2),
        "by_band": static_by_band,
    }

    return {
        "hospital": hospital.replace({np.nan: None}).to_dict(orient="records"),
        "department": department.replace({np.nan: None}).to_dict(orient="records"),
        "static_plan": static_plan,
        "cohort_plan": latest_cohort_resource_plan,
    }

@app.post("/api/cohort")
async def cohort(file: UploadFile = File(...), horizon_days: float = Form(DEFAULT_HORIZON_DAYS)):
    global latest_cohort_resource_plan
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        content = await file.read()
        if not content or len(content.strip()) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV file contains no patient rows.")
        normalized_df = normalize_input_df(df)
        missing = [c for c in RAW_COLUMNS if c not in normalized_df.columns]
        if missing:
            raise HTTPException(status_code=400, detail={"missing_columns": missing})
        probs = predict_df(normalized_df)
        results = normalized_df.copy()
        results["predicted_probability"] = probs
        results["risk_band"] = [risk_band(p) for p in probs]
        results["recommended_intervention"] = [INTERVENTION_POLICY[b]["intervention"] for b in results.risk_band]
        results["follow_up_within_days"] = [INTERVENTION_POLICY[b]["follow_up_within_days"] for b in results.risk_band]
        results = results.sort_values("predicted_probability", ascending=False)
        counts = results["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)

        resource_plan = compute_resource_plan(results, horizon_days=horizon_days)
        latest_cohort_resource_plan = resource_plan

        predictions_list = results.replace({np.nan: None}).to_dict(orient="records")

        return {
            "total": len(results),
            "counts": counts.to_dict(),
            "predictions": predictions_list,
            "resource_plan": resource_plan,
            "rows": predictions_list,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/cohort/template")
def cohort_template():
    df = pd.DataFrame(columns=RAW_COLUMNS)
    content = df.to_csv(index=False).encode()
    return StreamingResponse(io.BytesIO(content), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=cohort_template.csv"})
