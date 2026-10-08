
# 🏥 Clinical Risk Stratification for Patient Rehospitalization

An integrated machine learning system for predicting the risk of hospital readmission among diabetic patients. The project uses patient medical and hospital-record data to classify patients into **Low, Medium, and High readmission-risk categories** and provides insights that can support hospital resource planning.

## 📌 Project Overview

Hospital readmissions can increase healthcare costs and place additional pressure on hospital resources. Early identification of patients who are more likely to be readmitted can help healthcare professionals prioritize monitoring and follow-up care.

This project develops a machine learning-based **Clinical Readmission Risk Prediction System** using historical patient data.

The system:

* Processes patient and hospital admission data
* Performs data preprocessing and feature encoding
* Predicts readmission risk
* Categorizes patients into Low, Medium, and High risk
* Provides hospital resource-planning insights
* Provides a web interface for interacting with the prediction system
* Includes a REST API for connecting the machine learning model with the frontend

---

## 🎯 Objectives

* Predict the likelihood of hospital readmission.
* Identify patients belonging to different risk categories.
* Compare multiple machine learning algorithms.
* Provide an easy-to-use interface for predictions.
* Support hospital resource planning using predicted risk levels.
* Integrate machine learning with a web-based application.

---

## 🧠 Machine Learning Models

The project uses and compares multiple classification algorithms:

### 1. Logistic Regression

Used as a baseline classification model for predicting readmission risk.

### 2. Random Forest

An ensemble learning algorithm used to capture complex relationships between patient features.

### 3. XGBoost

A gradient-boosting algorithm used to improve prediction performance and handle complex feature relationships.

The final model can be selected based on the evaluation results obtained during experimentation.

---

## 📊 Risk Classification

The system groups patients into three risk categories:

| Risk Level | Meaning                             |
| ---------- | ----------------------------------- |
| 🟢 Low     | Lower predicted risk of readmission |
| 🟡 Medium  | Moderate predicted risk             |
| 🔴 High    | Higher predicted risk               |

These categories can help healthcare staff prioritize patients for appropriate follow-up and monitoring.

---

## 🏥 Hospital Resource Planning

The predicted risk information can also be used to provide an overview of possible resource requirements.

The project considers factors such as:

* Hospital beds
* Occupied beds
* ICU availability
* Nursing requirements
* Patient risk distribution

This allows the prediction system to be extended beyond individual patient prediction toward basic hospital resource planning.

---

## 🗂️ Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset.

The dataset contains hospital admission information for diabetic patients, including demographic information, medical history, medications, admission details, and other clinical attributes.

Additional mapping information is provided through:

```text
IDS_mapping.csv
```

The main patient dataset is:

```text
diabetic_data.csv
```

---

## 🔄 System Workflow

```text
Patient / Hospital Data
          ↓
Data Preprocessing
          ↓
Feature Encoding
          ↓
Train ML Models
          ↓
Model Evaluation
          ↓
Risk Prediction
          ↓
Low / Medium / High Risk
          ↓
Hospital Resource Insights
```

---

## 🏗️ Project Architecture

```text
                ┌─────────────────────┐
                │    User / Staff     │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   React Frontend    │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │    REST API         │
                │      FastAPI        │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ ML Prediction Model │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Risk Classification │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Resource Insights   │
                └─────────────────────┘
```

---

## 🛠️ Technologies Used

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Backend

* Python
* FastAPI
* Uvicorn

### Frontend

* React
* Vite
* JavaScript
* HTML
* CSS

### Development Tools

* VS Code
* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
Clinical_Readmission_Risk_Integrated_Finalized/
│
├── backend/
│   │
│   ├── ML_PROJECT_FINAL/
│   │   ├── api.py
│   │   ├── app.py
│   │   ├── diabetic_data.csv
│   │   ├── IDS_mapping.csv
│   │   └── ...
│   │
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vite.config.*
│   └── ...
│
├── .gitignore
└── README.md
```

---

# 🚀 Installation & Setup

## Prerequisites

Make sure the following are installed:

* Python 3.x
* Node.js
* npm
* Git

Check the installations:

```bash
python --version
node --version
npm --version
git --version
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/jerisham/ML_PBL.git
```

Move into the project:

```bash
cd ML_PBL
```

---

# 🐍 2. Backend Setup

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Navigate to the ML project:

```bash
cd ML_PROJECT_FINAL
```

---

# ⚡ 3. Start the FastAPI Backend

Run:

```bash
uvicorn api:app --reload
```

The backend will normally be available at:

```text
http://localhost:8000
```

FastAPI interactive API documentation:

```text
http://localhost:8000/docs
```

Keep this terminal running.

---

# 💻 4. Frontend Setup

Open a **new terminal**.

Navigate to the frontend:

```bash
cd frontend
```

Install the required packages:

```bash
npm install
```

If an environment example file is provided:

```powershell
copy .env.example .env
```

Configure the API URL:

```env
VITE_API_URL=http://localhost:8000
```

Start the frontend:

```bash
npm run dev
```

The application will normally be available at:

```text
http://localhost:5173
```

---

# 📊 5. Streamlit Version

The project also contains a Streamlit application.

From:

```text
backend/ML_PROJECT_FINAL
```

run:

```bash
streamlit run app.py
```

This launches the Streamlit-based version of the application.

---

## 🔌 API

The FastAPI backend provides endpoints that allow the frontend to communicate with the machine learning system.

Example architecture:

```text
React
  ↓
HTTP Request
  ↓
FastAPI
  ↓
ML Model
  ↓
Prediction
  ↓
JSON Response
  ↓
React
```

API documentation can be viewed through:

```text
http://localhost:8000/docs
```

---

## 📈 Expected Output

The system provides:

* Patient readmission-risk prediction
* Risk category
* Model prediction results
* Patient-level insights
* Risk distribution
* Basic hospital resource-planning information

Example:

```text
Patient
   ↓
Prediction
   ↓
Risk Level: HIGH
   ↓
Priority Monitoring
```

---

## 🔐 Data & Security

The project is developed as an academic machine learning application.

For real-world healthcare deployment, additional requirements would be necessary, including:

* Patient-data privacy
* Secure authentication
* Authorization
* Encryption
* Healthcare regulations
* Clinical validation
* Secure infrastructure

The dataset used for development is historical and should not be treated as a substitute for professional clinical judgment.

---

## ⚠️ Limitations

* Prediction quality depends on the available historical data.
* Historical datasets may contain missing or inconsistent values.
* Model performance depends on feature quality and preprocessing.
* The system has not been validated for real-world clinical deployment.
* Large-scale enterprise load testing has not been performed.
* Predictions should not replace decisions made by qualified healthcare professionals.
* Additional clinical validation would be required before real-world use.

---

## 🔮 Future Scope

Possible future improvements include:

* Integration with real-time hospital systems
* Larger and more diverse clinical datasets
* Advanced deep-learning models
* Explainable AI for individual predictions
* Real-time patient monitoring
* Improved hospital resource forecasting
* Automated alerts for high-risk patients
* Integration with Electronic Health Records
* Cloud deployment and scalable infrastructure
* Continuous model monitoring and retraining

---

## 👥 Project Team

**V S Mayuri:** https://github.com/vsmayuri08

**Jerisha M:**https://github.com/jerisham

Developed as an academic machine learning project.

---

## 📜 Disclaimer

This project is intended for **educational and research purposes only**.

The predictions generated by this system are not medical diagnoses and should not be used as a replacement for professional medical advice or clinical decision-making.
