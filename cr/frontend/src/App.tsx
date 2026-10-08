import React, { useEffect, useMemo, useRef, useState } from 'react'
import { BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const API = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000'
const ACCENT = '#0F6E5C', MUTED = '#5B655E', BORDER = '#DBDFD7'
const RISK: Record<string,string> = { Low:'#2F7D4F', Medium:'#A9711F', High:'#A6392E' }

type Options = {
  age_bands:string[]; medical_specialties:string[];
  admission_types:{id:number,label:string}[];
  discharge_dispositions:{id:number,label:string}[];
  admission_sources:{id:number,label:string}[];
  glucose:string[]; a1c:string[];
}
type Form = {
  gender:string; age_band:string; medical_specialty:string;
  admission_type_id:number; discharge_disposition_id:number; admission_source_id:number;
  time_in_hospital:number; num_lab_procedures:number; num_procedures:number; num_medications:number;
  number_outpatient:number; number_emergency:number; number_inpatient:number; number_diagnoses:number;
  max_glu_serum:string; A1Cresult:string; change:string; diabetesMed:string;
  num_meds_active:number; num_meds_changed:number; diag_1:string; diag_2:string; diag_3:string;
}
const defaults:Form = {
 gender:'Female', age_band:'[60-70)', medical_specialty:'InternalMedicine',
 admission_type_id:1, discharge_disposition_id:1, admission_source_id:7,
 time_in_hospital:4,num_lab_procedures:45,num_procedures:1,num_medications:15,
 number_outpatient:0,number_emergency:0,number_inpatient:0,number_diagnoses:7,
 max_glu_serum:'None',A1Cresult:'None',change:'No',diabetesMed:'No',
 num_meds_active:1,num_meds_changed:0,diag_1:'428',diag_2:'250',diag_3:''
}

function Card({children, className='', style }:{children:React.ReactNode,className?:string,style?:React.CSSProperties}) {
 return <div className={`card ${className}`} style={style}>{children}</div>
}
function Label({children}:{children:React.ReactNode}) { return <label>{children}</label> }
function Input({value,onChange,type='number'}:{value:any,onChange:(v:any)=>void,type?:string}) {
 return <input type={type} value={value} onChange={e=>onChange(type==='number'?Number(e.target.value):e.target.value)} />
}
function Select({value,onChange,options}:{value:any,onChange:(v:any)=>void,options:{value:any,label:string}[]|string[]}) {
 return <select value={value} onChange={e=>onChange(e.target.value)}>{options.map((o:any)=> {
   const v=typeof o==='object'?o.value:o, l=typeof o==='object'?o.label:o
   return <option key={String(v)} value={v}>{l}</option>
 })}</select>
}
function Button({children,onClick,disabled=false}:{children:React.ReactNode,onClick?:()=>void,disabled?:boolean}) {
 return <button className="primary" onClick={onClick} disabled={disabled}>{children}</button>
}

function Predictor({options}:{options:Options}) {
 const [form,setForm]=useState<Form>(defaults), [result,setResult]=useState<any>(null), [loading,setLoading]=useState(false), [error,setError]=useState('')
 const set=(k:keyof Form)=>(v:any)=>setForm(f=>({...f,[k]:v}))
 const predict=async()=>{
   setLoading(true);setError('')
   try {
    const r=await fetch(`${API}/api/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(form)})
    const data=await r.json(); if(!r.ok) throw new Error(data.detail||'Prediction failed')
    setResult(data)
   } catch(e:any){setError(e.message)} finally{setLoading(false)}
 }
 const selectNum=(items:{id:number,label:string}[])=>items.map(x=>({value:x.id,label:`${x.id} — ${x.label}`}))
 return <div className="page">
  <h1>Patient Risk Predictor</h1><p className="subtitle">This form is connected to the trained XGBoost readmission model.</p>
  <Card><h3>Patient & Admission</h3><div className="grid3">
   <div><Label>Gender</Label><Select value={form.gender} onChange={set('gender')} options={['Female','Male','Unknown/Invalid']}/></div>
   <div><Label>Age band</Label><Select value={form.age_band} onChange={set('age_band')} options={options.age_bands}/></div>
   <div><Label>Medical specialty</Label><Select value={form.medical_specialty} onChange={set('medical_specialty')} options={options.medical_specialties}/></div>
   <div><Label>Admission type</Label><Select value={form.admission_type_id} onChange={v=>set('admission_type_id')(Number(v))} options={selectNum(options.admission_types)}/></div>
   <div><Label>Discharge disposition</Label><Select value={form.discharge_disposition_id} onChange={v=>set('discharge_disposition_id')(Number(v))} options={selectNum(options.discharge_dispositions)}/></div>
   <div><Label>Admission source</Label><Select value={form.admission_source_id} onChange={v=>set('admission_source_id')(Number(v))} options={selectNum(options.admission_sources)}/></div>
  </div></Card>
  <Card><h3>Hospital Utilization</h3><div className="grid3">
   {([
    ['time_in_hospital','Time in hospital (days)'],['num_lab_procedures','Lab procedures'],['num_procedures','Procedures'],
    ['num_medications','Medications'],['number_outpatient','Outpatient visits (prior year)'],
    ['number_emergency','Emergency visits (prior year)'],['number_inpatient','Inpatient visits (prior year)'],['number_diagnoses','Diagnoses'],
    ['num_meds_active','Active diabetes medications'],['num_meds_changed','Medications changed']
   ] as [keyof Form,string][]).map(([k,l])=><div key={String(k)}><Label>{l}</Label><Input value={form[k]} onChange={set(k)}/></div>)}
  </div></Card>
  <Card><h3>Clinical Results & Diagnoses</h3><div className="grid3">
   <div><Label>Max glucose serum</Label><Select value={form.max_glu_serum} onChange={set('max_glu_serum')} options={options.glucose}/></div>
   <div><Label>A1C result</Label><Select value={form.A1Cresult} onChange={set('A1Cresult')} options={options.a1c}/></div>
   <div><Label>Medication changed</Label><Select value={form.change} onChange={set('change')} options={['No','Ch']}/></div>
   <div><Label>On diabetes medication</Label><Select value={form.diabetesMed} onChange={set('diabetesMed')} options={['No','Yes']}/></div>
   <div><Label>Primary ICD-9</Label><Input type="text" value={form.diag_1} onChange={set('diag_1')}/></div>
   <div><Label>Secondary ICD-9</Label><Input type="text" value={form.diag_2} onChange={set('diag_2')}/></div>
   <div><Label>Tertiary ICD-9</Label><Input type="text" value={form.diag_3} onChange={set('diag_3')}/></div>
  </div></Card>
  <div className="actions"><Button onClick={predict} disabled={loading}>{loading?'Scoring…':'Predict readmission risk'}</Button>
   <button className="link" onClick={()=>{setForm(defaults);setResult(null);setError('')}}>Clear</button></div>
  {error&&<div className="error">{error}</div>}
  {result&&<div className="result">
   <h2>Risk Assessment Result</h2>
   <div className="risk" style={{borderLeftColor:RISK[result.band]}}><small>Predicted 30-day readmission probability</small>
    <strong style={{color:RISK[result.band]}}>{result.probability.toFixed(1)}%</strong><small>Risk band</small><b style={{color:RISK[result.band]}}>{result.band}</b></div>
   <div className="grid2">
    <Card><h3>Increasing risk</h3>{result.increasing_risk.length?result.increasing_risk.map((x:string)=><p className="list" key={x}>› {x}</p>):<p>No strong risk-raising factors identified.</p>}</Card>
    <Card><h3>Lowering risk</h3>{result.lowering_risk.length?result.lowering_risk.map((x:string)=><p className="list" key={x}>› {x}</p>):<p>No strong risk-lowering factors identified.</p>}</Card>
   </div>
   <Card><h3>Recommended intervention</h3><p>{result.intervention}</p><p>Follow up within <b>{result.follow_up_within_days} days</b> · Assign <b>{result.staff_role}</b></p></Card>
  </div>}
 </div>
}

type Capacity = {
  total_beds: number;
  occupied_beds: number;
  total_icu_beds: number;
  available_icu_beds: number;
  nurses_available: number;
  doctors_available: number;
  case_managers_available: number;
  pharmacists_available: number;
  horizon_days: number;
}

const defaultCapacity: Capacity = {
  total_beds: 200,
  occupied_beds: 150,
  total_icu_beds: 30,
  available_icu_beds: 10,
  nurses_available: 20,
  doctors_available: 10,
  case_managers_available: 2,
  pharmacists_available: 3,
  horizon_days: 30,
}

function StatusBadge({ status }: { status: 'SHORTAGE' | 'NEAR CAPACITY' | 'SUFFICIENT' }) {
  const styles = {
    SHORTAGE: { bg: '#FDF2F2', border: '#F8B4B4', color: '#A6392E' },
    'NEAR CAPACITY': { bg: '#FDF8F0', border: '#FCD39D', color: '#A9711F' },
    SUFFICIENT: { bg: '#F2F9F4', border: '#A3E2B8', color: '#2F7D4F' },
  }[status]

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '3px 8px',
        borderRadius: '4px',
        fontSize: '11px',
        fontWeight: 600,
        fontFamily: 'var(--mono)',
        backgroundColor: styles.bg,
        border: `1px solid ${styles.border}`,
        color: styles.color,
      }}
    >
      {status}
    </span>
  )
}

function CapacityVsDemand({ plan, capacity, onCapacityChange }: { plan: any; capacity: Capacity; onCapacityChange: (c: Capacity) => void }) {
  const [showConfig, setShowConfig] = useState(false)
  if (!plan) return null

  const availableBeds = Math.max(0, capacity.total_beds - capacity.occupied_beds)
  const totalBedDays = plan.total_estimated_bed_days ?? 0
  const bedDemand = plan.total_avg_daily_beds_needed ?? plan.beds_required ?? (totalBedDays / Math.max(1, capacity.horizon_days))

  const bedStatus: 'SHORTAGE' | 'NEAR CAPACITY' | 'SUFFICIENT' =
    availableBeds < bedDemand ? 'SHORTAGE' : availableBeds <= bedDemand * 1.15 ? 'NEAR CAPACITY' : 'SUFFICIENT'
  const additionalBeds = availableBeds < bedDemand ? Math.ceil(bedDemand - availableBeds) : 0

  const icuReq = plan.icu_beds_required ?? Math.ceil(plan.avg_daily_icu_beds_needed ?? ((totalBedDays * 0.15) / Math.max(1, capacity.horizon_days)))
  const icuAvail = capacity.available_icu_beds
  const icuStatus: 'SHORTAGE' | 'NEAR CAPACITY' | 'SUFFICIENT' =
    icuAvail < icuReq ? 'SHORTAGE' : icuAvail === icuReq ? 'NEAR CAPACITY' : 'SUFFICIENT'
  const icuAdd = Math.max(0, icuReq - icuAvail)

  const cmReq = plan.staff_required?.case_managers ?? plan.by_band?.High?.staff_required ?? 0
  const cmAvail = capacity.case_managers_available
  const cmStatus: 'SHORTAGE' | 'NEAR CAPACITY' | 'SUFFICIENT' =
    cmAvail < cmReq ? 'SHORTAGE' : cmAvail === cmReq ? 'NEAR CAPACITY' : 'SUFFICIENT'
  const cmAdd = Math.max(0, cmReq - cmAvail)

  const nurseReq = plan.staff_required?.follow_up_nurses ?? plan.by_band?.Medium?.staff_required ?? 0
  const nurseAvail = capacity.nurses_available
  const nurseStatus: 'SHORTAGE' | 'NEAR CAPACITY' | 'SUFFICIENT' =
    nurseAvail < nurseReq ? 'SHORTAGE' : nurseAvail === nurseReq ? 'NEAR CAPACITY' : 'SUFFICIENT'
  const nurseAdd = Math.max(0, nurseReq - nurseAvail)

  const updateCap = (k: keyof Capacity) => (v: number) => onCapacityChange({ ...capacity, [k]: v })

  const summaryRows = [
    {
      resource: 'Hospital Beds (Daily Demand)',
      demand: `${bedDemand.toFixed(1)} beds/day`,
      available: `${availableBeds} beds`,
      diff: availableBeds < bedDemand ? `-${additionalBeds} beds` : `+${(availableBeds - bedDemand).toFixed(1)} beds`,
      status: bedStatus,
    },
    {
      resource: 'ICU Beds (Daily Demand)',
      demand: `${icuReq} beds`,
      available: `${icuAvail} beds`,
      diff: icuAvail < icuReq ? `-${icuAdd} beds` : `+${icuAvail - icuReq} beds`,
      status: icuStatus,
    },
    {
      resource: 'Case Managers (High Risk)',
      demand: `${cmReq} staff`,
      available: `${cmAvail} staff`,
      diff: cmAvail < cmReq ? `-${cmAdd} staff` : `+${cmAvail - cmReq} staff`,
      status: cmStatus,
    },
    {
      resource: 'Follow-up Staff (Medium Risk)',
      demand: `${nurseReq} staff`,
      available: `${nurseAvail} staff`,
      diff: nurseAvail < nurseReq ? `-${nurseAdd} staff` : `+${nurseAvail - nurseReq} staff`,
      status: nurseStatus,
    },
  ]

  return (
    <Card style={{ marginBottom: '22px' }}>
      <div className="row" style={{ marginBottom: '10px' }}>
        <div>
          <h2 style={{ margin: 0 }}>Hospital Capacity vs Demand</h2>
          <p className="subtitle" style={{ margin: 0 }}>
            Operational comparison based on predicted ML cohort demand and user-configured capacity.
          </p>
        </div>
        <button className="primary" style={{ padding: '6px 12px', fontSize: '12px' }} onClick={() => setShowConfig(!showConfig)}>
          {showConfig ? 'Hide Capacity Setup' : '⚙ Configure Capacity Inputs'}
        </button>
      </div>

      {showConfig && (
        <div style={{ background: '#FAFBF9', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px', marginBottom: '18px' }}>
          <h3 style={{ marginTop: 0, marginBottom: '12px' }}>Hospital Capacity Inputs</h3>
          <div className="grid3">
            <div><Label>Total Hospital Beds</Label><Input value={capacity.total_beds} onChange={updateCap('total_beds')} /></div>
            <div><Label>Currently Occupied Beds</Label><Input value={capacity.occupied_beds} onChange={updateCap('occupied_beds')} /></div>
            <div><Label>Available Beds (Calculated)</Label><input disabled value={availableBeds} style={{ background: '#eef1ed' }} /></div>
            <div><Label>Planning horizon (days)</Label><Input value={capacity.horizon_days} onChange={updateCap('horizon_days')} /></div>
            <div><Label>Total ICU Beds</Label><Input value={capacity.total_icu_beds} onChange={updateCap('total_icu_beds')} /></div>
            <div><Label>Available ICU Beds</Label><Input value={capacity.available_icu_beds} onChange={updateCap('available_icu_beds')} /></div>
            <div><Label>Nurses Available</Label><Input value={capacity.nurses_available} onChange={updateCap('nurses_available')} /></div>
            <div><Label>Doctors Available</Label><Input value={capacity.doctors_available} onChange={updateCap('doctors_available')} /></div>
            <div><Label>Case Managers Available</Label><Input value={capacity.case_managers_available} onChange={updateCap('case_managers_available')} /></div>
            <div><Label>Pharmacists Available</Label><Input value={capacity.pharmacists_available} onChange={updateCap('pharmacists_available')} /></div>
          </div>
          <p className="hint" style={{ marginTop: '10px', fontSize: '12px' }}>
            <strong>Planning horizon:</strong> how many days this patient group is assumed to span (e.g. "this cohort is one week of discharges" → 7).
            Demand totals are divided by this to get a daily figure comparable to your beds/staff on hand.
          </p>
        </div>
      )}

      <div className="grid3" style={{ marginBottom: '16px' }}>
        <div style={{ background: '#fff', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px' }}>
          <div className="row" style={{ marginBottom: '6px' }}>
            <h3 style={{ margin: 0 }}>Bed Capacity</h3>
            <StatusBadge status={bedStatus} />
          </div>
          <p style={{ margin: '4px 0' }}>Available beds: <strong>{availableBeds}</strong></p>
          <p style={{ margin: '4px 0' }}>Estimated demand: <strong>{bedDemand.toFixed(1)} beds/day</strong> <span className="hint" style={{ fontSize: '11px' }}>({totalBedDays.toFixed(1)} bed-days over {capacity.horizon_days} days)</span></p>
          {additionalBeds > 0 ? (
            <p style={{ color: '#A6392E', margin: '6px 0 0', fontWeight: 600 }}>Additional beds required: {additionalBeds}</p>
          ) : (
            <p style={{ color: '#2F7D4F', margin: '6px 0 0' }}>Surplus capacity: {(availableBeds - bedDemand).toFixed(1)} beds</p>
          )}
        </div>

        <div style={{ background: '#fff', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px' }}>
          <div className="row" style={{ marginBottom: '6px' }}>
            <h3 style={{ margin: 0 }}>ICU Bed Capacity</h3>
            <StatusBadge status={icuStatus} />
          </div>
          <p style={{ margin: '4px 0' }}>Available ICU beds: <strong>{icuAvail}</strong></p>
          <p style={{ margin: '4px 0' }}>Estimated demand: <strong>{icuReq} beds/day</strong> <span className="hint" style={{ fontSize: '11px' }}>(prototype acuity est.)</span></p>
          {icuAdd > 0 ? (
            <p style={{ color: '#A6392E', margin: '6px 0 0', fontWeight: 600 }}>Additional ICU beds required: {icuAdd}</p>
          ) : (
            <p style={{ color: '#2F7D4F', margin: '6px 0 0' }}>Surplus ICU capacity: {icuAvail - icuReq} beds</p>
          )}
        </div>

        <div style={{ background: '#fff', border: '1px solid var(--border)', borderRadius: '8px', padding: '16px' }}>
          <div className="row" style={{ marginBottom: '6px' }}>
            <h3 style={{ margin: 0 }}>Case Managers</h3>
            <StatusBadge status={cmStatus} />
          </div>
          <p style={{ margin: '4px 0' }}>Required: <strong>{cmReq}</strong></p>
          <p style={{ margin: '4px 0' }}>Available: <strong>{cmAvail}</strong></p>
          {cmAdd > 0 ? (
            <p style={{ color: '#A6392E', margin: '6px 0 0', fontWeight: 600 }}>Additional required: {cmAdd}</p>
          ) : (
            <p style={{ color: '#2F7D4F', margin: '6px 0 0' }}>Surplus staff: {cmAvail - cmReq}</p>
          )}
        </div>
      </div>

      <DataTable rows={summaryRows} />
    </Card>
  )
}

function CohortResourceForecast({ plan, capacity, onCapacityChange }: { plan: any; capacity: Capacity; onCapacityChange: (c: Capacity) => void }) {
  const [showAssumptions, setShowAssumptions] = useState(false)
  if (!plan || !plan.by_band) return null
  const { High, Medium, Low } = plan.by_band
  const horizon = plan.planning_horizon ?? plan.horizon_days ?? capacity.horizon_days
  const totalPts = plan.total_patients ?? 0
  const highRiskPts = plan.predicted_high_risk_patients ?? High?.n_patients ?? 0
  const medRiskPts = plan.predicted_medium_risk_patients ?? Medium?.n_patients ?? 0
  const lowRiskPts = plan.predicted_low_risk_patients ?? Low?.n_patients ?? 0
  const expectedReadmissions = plan.expected_readmissions ?? plan.predicted_readmissions ?? plan.total_expected_readmissions ?? 0
  const avgProb = plan.average_predicted_probability !== undefined
    ? `${plan.average_predicted_probability}%`
    : totalPts > 0 ? `${((expectedReadmissions / totalPts) * 100).toFixed(1)}%` : '0.0%'
  const avgDailyBeds = plan.beds_required ?? plan.total_avg_daily_beds_needed ?? 0
  const icuBeds = plan.icu_beds_required ?? Math.ceil(plan.avg_daily_icu_beds_needed ?? 0)

  return (
    <div style={{ marginBottom: '22px' }}>
      <Card>
        <h2>Resource Planning Based on Uploaded Cohort</h2>
        <p className="subtitle" style={{ marginBottom: '10px' }}>
          Calculated dynamically from XGBoost ML model predictions for current uploaded cohort (Total: {totalPts} Patients, Planning Horizon: {horizon} Days)
        </p>

        <div style={{ background: '#F0F4F2', border: '1px solid #C2D4CC', borderRadius: '6px', padding: '10px 14px', marginBottom: '14px', fontSize: '13px', color: '#1B211D' }}>
          <strong>💡 Key Clinical Distinction:</strong> <strong>High-Risk Patients ({highRiskPts})</strong> represents patients with predicted readmission probability ≥ 66%. 
          <strong> Expected Readmissions ({expectedReadmissions})</strong> is the probabilistic expected total readmissions across all patients (SUM of probabilities).
        </div>

        <div className="metrics" style={{ flexWrap: 'wrap', gap: '12px' }}>
          <Metric label="Total Uploaded Patients" value={totalPts} />
          <Metric label="Predicted High Risk (p≥66%)" value={highRiskPts} color={RISK.High} />
          <Metric label="Medium Risk (33-65%)" value={medRiskPts} color={RISK.Medium} />
          <Metric label="Low Risk (p<33%)" value={lowRiskPts} color={RISK.Low} />
          <Metric label="Avg Readmission Prob" value={avgProb} />
          <Metric label="Expected Readmissions" value={expectedReadmissions} decimal />
          <Metric label="Est Bed Demand (Beds/Day)" value={avgDailyBeds} decimal />
          <Metric label="Est ICU Beds Needed" value={icuBeds} />
          <Metric label="Planning Horizon" value={`${horizon} days`} />
        </div>

        <div className="grid3" style={{ marginTop: '16px' }}>
          <div style={{ background: '#fff', border: '1px solid var(--border)', borderTop: `4px solid ${RISK.High}`, borderRadius: '8px', padding: '16px' }}>
            <h3 style={{ color: RISK.High, marginTop: 0, marginBottom: '10px' }}>High Risk Resources</h3>
            <p style={{ margin: '4px 0' }}>Patients (p ≥ 0.66): <strong>{highRiskPts}</strong></p>
            <p style={{ margin: '4px 0' }}>Expected readmissions: <strong>{High?.expected_readmissions ?? 0}</strong></p>
            <p style={{ margin: '4px 0' }}>Estimated bed-days: <strong>{High?.estimated_bed_days ?? 0}</strong> <span className="hint" style={{ fontSize: '11px' }}>({High?.avg_daily_beds_needed ?? 0} beds/day)</span></p>
            <p style={{ margin: '6px 0' }}>Case managers required: <strong style={{ font: '600 18px var(--mono)', color: RISK.High }}>{High?.staff_required ?? 0}</strong></p>
            <p className="hint" style={{ marginTop: '10px', fontSize: '12px' }}><strong>Follow-up workload:</strong> {High?.workload}</p>
          </div>

          <div style={{ background: '#fff', border: '1px solid var(--border)', borderTop: `4px solid ${RISK.Medium}`, borderRadius: '8px', padding: '16px' }}>
            <h3 style={{ color: RISK.Medium, marginTop: 0, marginBottom: '10px' }}>Medium Risk Resources</h3>
            <p style={{ margin: '4px 0' }}>Patients (0.33 ≤ p &lt; 0.66): <strong>{medRiskPts}</strong></p>
            <p style={{ margin: '4px 0' }}>Expected readmissions: <strong>{Medium?.expected_readmissions ?? 0}</strong></p>
            <p style={{ margin: '4px 0' }}>Estimated bed-days: <strong>{Medium?.estimated_bed_days ?? 0}</strong> <span className="hint" style={{ fontSize: '11px' }}>({Medium?.avg_daily_beds_needed ?? 0} beds/day)</span></p>
            <p style={{ margin: '6px 0' }}>Follow-up staff required: <strong style={{ font: '600 18px var(--mono)', color: RISK.Medium }}>{Medium?.staff_required ?? 0}</strong></p>
            <p className="hint" style={{ marginTop: '10px', fontSize: '12px' }}><strong>Follow-up workload:</strong> {Medium?.workload}</p>
          </div>

          <div style={{ background: '#fff', border: '1px solid var(--border)', borderTop: `4px solid ${RISK.Low}`, borderRadius: '8px', padding: '16px' }}>
            <h3 style={{ color: RISK.Low, marginTop: 0, marginBottom: '10px' }}>Low Risk Resources</h3>
            <p style={{ margin: '4px 0' }}>Patients (p &lt; 0.33): <strong>{lowRiskPts}</strong></p>
            <p style={{ margin: '4px 0' }}>Expected readmissions: <strong>{Low?.expected_readmissions ?? 0}</strong></p>
            <p style={{ margin: '4px 0' }}>Estimated bed-days: <strong>{Low?.estimated_bed_days ?? 0}</strong> <span className="hint" style={{ fontSize: '11px' }}>({Low?.avg_daily_beds_needed ?? 0} beds/day)</span></p>
            <p className="hint" style={{ marginTop: '10px', fontSize: '12px' }}><strong>Standard follow-up:</strong> {Low?.workload}</p>
          </div>
        </div>

        {plan.assumptions && plan.assumptions.length > 0 && (
          <div style={{ marginTop: '16px' }}>
            <button className="link" onClick={() => setShowAssumptions(!showAssumptions)} style={{ fontSize: '13px', fontWeight: 600 }}>
              {showAssumptions ? '▲ Hide Resource Calculation Assumptions' : '▼ View Prototype Resource Calculation Assumptions'}
            </button>
            {showAssumptions && (
              <div style={{ marginTop: '8px', padding: '12px', background: '#F8F9F7', border: '1px solid var(--border)', borderRadius: '6px', fontSize: '12px' }}>
                <strong style={{ display: 'block', marginBottom: '6px' }}>Documented Planning Assumptions & Formulas:</strong>
                <ul style={{ margin: 0, paddingLeft: '20px' }}>
                  {plan.assumptions.map((asm: string, idx: number) => (
                    <li key={idx} style={{ marginBottom: '4px' }}>{asm}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </Card>

      <CapacityVsDemand plan={plan} capacity={capacity} onCapacityChange={onCapacityChange} />
    </div>
  )
}

function Resource({ capacity, onCapacityChange }: { capacity: Capacity; onCapacityChange: (c: Capacity) => void }) {
  const [data, setData] = useState<any>(null), [error, setError] = useState('')
  useEffect(() => { fetch(`${API}/api/resource`).then(r => r.json()).then(setData).catch(e => setError(e.message)) }, [])
  if (error) return <div className="page"><div className="error">{error}</div></div>
  if (!data) return <div className="page"><p>Loading resource plan…</p></div>
  const cohortPlan = data.cohort_plan
  const hospital = cohortPlan?.hospital || data.hospital || []
  const department = cohortPlan?.department || data.department || []

  return (
    <div className="page">
      <h1>Hospital Resource Plan</h1>
      <p className="subtitle">
        {cohortPlan
          ? `Dynamic forecast derived from the current uploaded patient cohort (${cohortPlan.total_patients} patients).`
          : 'Live data loaded from hospital resource planning calculations.'}
      </p>
      {cohortPlan ? (
        <CohortResourceForecast plan={cohortPlan} capacity={capacity} onCapacityChange={onCapacityChange} />
      ) : data.static_plan ? (
        <>
          <p className="hint" style={{ margin: '0 0 12px' }}>
            This reference plan is built from the full historical evaluation set (many hospitals, several years) — it's illustrative, not a forecast for one hospital's live patient load.
            Upload a real cohort under Batch / Cohort Prioritization for an actual capacity comparison.
          </p>
          <CapacityVsDemand plan={data.static_plan} capacity={capacity} onCapacityChange={onCapacityChange} />
        </>
      ) : null}
      <Card><h3>Hospital resource plan by risk band {cohortPlan ? '(Current Cohort)' : ''}</h3><DataTable rows={hospital} /></Card>
      <Card><h3>Department-level allocation {cohortPlan ? '(Current Cohort)' : ''}</h3><DataTable rows={department} /></Card>
      {department.length > 0 && (
        <Card>
          <h3>Projected Department Bed Demand</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={department.slice(0, 10).map((r: any) => ({ name: String(r.medical_specialty || r.department || '').slice(0, 16), bed_days: Number(r.expected_bed_days || 0) }))}>
              <CartesianGrid vertical={false} stroke={BORDER} />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="bed_days" fill={ACCENT} name="Expected Bed-Days" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}
    </div>
  )
}

function DataTable({ rows }: { rows: any[] }) {
  if (!rows.length) return <p>No data.</p>
  const cols = Object.keys(rows[0])
  return <div className="tablewrap"><table><thead><tr>{cols.map(c => <th key={c}>{c.replace(/_/g, ' ')}</th>)}</tr></thead>
    <tbody>{rows.map((r, i) => <tr key={i}>{cols.map(c => <td key={c}>{typeof r[c] === 'number' ? Number(r[c]).toFixed(2).replace(/\.00$/, '') : String(r[c] ?? '')}</td>)}</tr>)}</tbody></table></div>
}

function Batch({ capacity, onCapacityChange }: { capacity: Capacity; onCapacityChange: (c: Capacity) => void }) {
  const [result, setResult] = useState<any>(null), [loading, setLoading] = useState(false), [error, setError] = useState('')
  const ref = useRef<HTMLInputElement>(null)
  const [selectedHorizon, setSelectedHorizon] = useState<number>(capacity.horizon_days)

  const upload = async (file: File, horizonOverride?: number) => {
    setLoading(true); setError('')
    const horizonToUse = horizonOverride ?? selectedHorizon
    const fd = new FormData(); fd.append('file', file); fd.append('horizon_days', String(horizonToUse))
    try {
      const r = await fetch(`${API}/api/cohort`, { method: 'POST', body: fd })
      const d = await r.json()
      if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail))
      setResult(d)
    } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }

  const handleHorizonChange = (newHorizon: number) => {
    setSelectedHorizon(newHorizon)
    onCapacityChange({ ...capacity, horizon_days: newHorizon })
    if (ref.current?.files?.[0]) {
      upload(ref.current.files[0], newHorizon)
    }
  }

  const download = () => {
    if (!result) return;
    const patientRows = result.predictions || result.rows || []
    if (!patientRows.length) return;
    const cols = Object.keys(patientRows[0] || {})
    const csv = [cols.join(','), ...patientRows.map((r: any) => cols.map(c => JSON.stringify(r[c] ?? '')).join(','))].join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = 'prioritized_readmission_risk_list.csv'
    a.click()
  }

  const chart = useMemo(() => result ? ['High', 'Medium', 'Low'].map(b => ({ band: b, count: result.counts[b] || 0 })) : [], [result])

  return (
    <div className="page">
      <h1>Batch / Cohort Prioritization</h1>
      <p className="subtitle">Upload a cohort CSV to score all patients and dynamically generate hospital resource forecasts.</p>
      <Card>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '16px', marginBottom: '12px' }}>
          <input ref={ref} hidden type="file" accept=".csv" onChange={e => e.target.files?.[0] && upload(e.target.files[0])} />
          <Button onClick={() => ref.current?.click()} disabled={loading}>{loading ? 'Scoring cohort…' : 'Upload cohort CSV'}</Button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Label>Planning Horizon:</Label>
            <Select
              value={selectedHorizon}
              onChange={v => handleHorizonChange(Number(v))}
              options={[
                { value: 7, label: '7 Days (One Week)' },
                { value: 14, label: '14 Days (Two Weeks)' },
                { value: 30, label: '30 Days (One Month)' },
              ]}
            />
          </div>
        </div>
        <p className="hint">The CSV must contain the model's required preprocessed input columns. Use the backend template if needed.</p>
        <p className="hint">Scored against a {selectedHorizon}-day planning horizon — resource metrics update automatically when changed.</p>
        <a className="link" href={`${API}/api/cohort/template`}>Download CSV template</a>
      </Card>
      {error && <div className="error">{error}</div>}
      {result && <>
        <CohortResourceForecast plan={result.resource_plan} capacity={capacity} onCapacityChange={onCapacityChange} />
        <Card><h3>Risk Band Distribution</h3><ResponsiveContainer width="100%" height={240}><BarChart data={chart}><CartesianGrid vertical={false} stroke={BORDER} /><XAxis dataKey="band" /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count">{chart.map(x => <Cell key={x.band} fill={RISK[x.band]} />)}</Bar></BarChart></ResponsiveContainer></Card>
        <Card><div className="row"><h3>Prioritized Patient List (Model Predictions)</h3><Button onClick={download}>Download CSV</Button></div><DataTable rows={result.predictions || result.rows || []} /></Card>
      </>}
    </div>
  )
}

function Metric({ label, value, color, decimal = false }: { label: string, value: number | string, color?: string, decimal?: boolean }) {
  const formatted = typeof value === 'number' ? (decimal ? value.toFixed(1) : value) : value
  return <div className="metric"><small>{label}</small><strong style={color ? { color } : {}}>{formatted}</strong></div>
}

export default function App() {
  const [tab, setTab] = useState('predictor'), [options, setOptions] = useState<Options | null>(null), [apiOk, setApiOk] = useState(false)
  const [capacity, setCapacity] = useState<Capacity>(defaultCapacity)

  useEffect(() => { Promise.all([fetch(`${API}/api/health`), fetch(`${API}/api/options`)]).then(async ([h, o]) => { setApiOk(h.ok); setOptions(await o.json()) }).catch(() => setApiOk(false)) }, [])
  return <><header><div className="headerInner"><div className="brand">Clinical Readmission Risk & Hospital Resource Planning <span className={apiOk ? 'ok' : 'offline'}>{apiOk ? '● API connected' : '● API offline'}</span></div>
    <nav>{[['predictor', 'Patient Risk Predictor'], ['resource', 'Hospital Resource Plan'], ['batch', 'Batch / Cohort Prioritization']].map(([id, l]) => <button className={tab === id ? 'active' : ''} key={id} onClick={() => setTab(id)}>{l}</button>)}</nav></div></header>
    <main>{tab === 'predictor' && options ? <Predictor options={options} /> : tab === 'predictor' ? <div className="page"><p>Connecting to backend…</p></div> : tab === 'resource' ? <Resource capacity={capacity} onCapacityChange={setCapacity} /> : <Batch capacity={capacity} onCapacityChange={setCapacity} />}</main></>
}


