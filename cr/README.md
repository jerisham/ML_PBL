# Clinical Readmission Risk & Hospital Resource Planning

This version integrates the React/Vite frontend with the trained XGBoost backend.

**Flow: Predict → Resource Plan.** Score a patient (or a whole cohort) for 30-day readmission
risk first; the resource plan (beds, staffing) is derived from those predictions.

## Run backend
```bash
cd backend/ML_PROJECT_FINAL
pip install -r ../requirements.txt
uvicorn api:app --reload --port 8000
```

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://localhost:5173).

The frontend calls:
- `GET /api/health`
- `GET /api/options`
- `POST /api/predict`
- `GET /api/resource`
- `POST /api/cohort` (accepts an optional `horizon_days` field, default 30)
- `GET /api/cohort/template`

The old Streamlit `app.py` is retained as the original reference implementation; `api.py` is the integration layer used by the React frontend.

## Fixes in this version

- **Bed/staffing calculation was comparing mismatched units.** `expected_bed_days` (a cumulative
  total across the whole cohort, with no time period attached) was being compared directly
  against a snapshot "beds available" count, so the dashboard reported a shortage almost
  regardless of actual capacity. Fixed by introducing an explicit, adjustable "planning horizon
  (days)" — the number of days a cohort is assumed to span — and deriving:
  - `avg_daily_beds_needed = expected_bed_days / horizon_days` (comparable to a beds count)
  - `concurrent_caseload` via Little's Law (arrival rate × how long each patient stays in the
    follow-up program), replacing a flawed "total headcount / caseload ratio" staffing formula.
  Applied consistently in the notebook (Section 8.2/8.3), `api.py`, and the frontend.
- The static `hospital_resource_plan.csv` / `department_resource_plan.csv` are built from the
  full historical evaluation set (many hospitals, several years), not a single hospital's live
  patient load — treat them as an illustrative reference. For a real capacity comparison, upload
  an actual cohort via **Batch / Cohort Prioritization** and set the horizon to what that cohort
  actually spans (e.g. "one week of discharges" → 7 days).
- Fixed a JS syntax error in `App.tsx` (missing closing `}`) that prevented the frontend from
  compiling at all.
- Fixed a type error: `Card` didn't accept the `style` prop it was being passed.
- Removed a hardcoded placeholder ("186.6 bed-days", fixed staff counts) that was shown whenever
  no cohort had been uploaded; replaced with a real plan computed from the static CSVs.

