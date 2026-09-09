# 🩺 Stroke Prediction Project

A machine learning-based web application for predicting the risk of stroke based on patient health and demographic information.

The system provides an interactive web interface that allows users to enter patient information and receive a predicted probability of stroke and a corresponding risk assessment.

---

## 📌 Project Overview

Stroke is a serious medical condition that can lead to long-term disability or death. Early identification of individuals who may be at higher risk of stroke can support timely medical assessment and preventive care.

This project aims to build an end-to-end stroke prediction system using machine learning models and deploy the system as a web application.

The application consists of:

- A **FastAPI backend** for model inference and API services.
- A **Streamlit frontend** for user interaction and visualization.
- Multiple trained machine learning models for stroke prediction.
- Supporting artifacts, model thresholds, and statistical data.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Develop a machine learning system for stroke risk prediction.
- Build an API for model inference using FastAPI.
- Develop an interactive web interface using Streamlit.
- Allow users to select different machine learning models.
- Provide the predicted probability of stroke.
- Provide an understandable risk assessment for users.
- Organize the project into a complete and reusable application structure.

---

## 🤖 Machine Learning Models

The system supports the following machine learning models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Users can select a model directly from the web interface when making a prediction.

---

## 🏗️ System Architecture

The system consists of two main components:

```text
                    ┌────────────────────────┐
                    │      User Input        │
                    │   Streamlit Frontend   │
                    └────────────┬───────────┘
                                 │
                                 │ HTTP Request
                                 ▼
                    ┌────────────────────────┐
                    │     FastAPI Backend    │
                    │       /predict         │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Machine Learning     │
                    │        Models           │
                    │                         │
                    │ Logistic Regression    │
                    │ Random Forest           │
                    │ XGBoost                 │
                    │ LightGBM                │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Prediction Result    │
                    │ Probability + Risk      │
                    └────────────────────────┘
```

---

## 🔄 Prediction Workflow

```text
Patient Information
        ↓
Streamlit Frontend
        ↓
POST /predict
        ↓
FastAPI Backend
        ↓
Input Validation
        ↓
Feature Processing
        ↓
Selected Machine Learning Model
        ↓
Stroke Probability
        ↓
Risk Assessment
        ↓
Display Result
```

---

## 📁 Project Structure

```text
STROKE PREDICTION PROJECTS/
│
├── Backend/
│   ├── Artifacts/
│   ├── List/
│   ├── Models/
│   ├── Statistics_data/
│   ├── Threshold/
│   └── main.py
│
├── Frontend/
│   └── app.py
│
├── Output/
│   ├── cv_summary_results.csv
│   ├── results_summary.pkl
│   ├── search_results.pkl
│   └── stroke_data_clean.csv
│
├── .gitignore
├── README.md
└── Stroke_Prediction.ipynb
```

### Backend

The `Backend/` directory contains the FastAPI application and resources required for prediction.

#### `main.py`

The main FastAPI application responsible for:

- Loading trained machine learning models.
- Loading model-specific thresholds.
- Validating prediction requests.
- Processing patient input.
- Generating stroke probabilities.
- Returning prediction results.
- Providing metadata and statistical information to the frontend.

#### `Models/`

Contains the trained machine learning models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

#### `Threshold/`

Contains the thresholds used by the prediction models.

#### `Statistics_data/`

Contains statistical data used by the frontend application.

#### `Artifacts/`

Contains supporting machine learning artifacts used by the application.

#### `List/`

Contains supporting lists and resources used by the backend.

---

### Frontend

The `Frontend/` directory contains the Streamlit application.

#### `app.py`

The main user interface of the project.

The frontend provides:

- Stroke rate analytics.
- Patient information input.
- Machine learning model selection.
- Stroke probability prediction.
- Risk assessment.
- Prediction date.
- Model information.
- Medical disclaimer.

---

### Output

The `Output/` directory contains generated datasets and machine learning results.

```text
Output/
├── cv_summary_results.csv
├── results_summary.pkl
├── search_results.pkl
└── stroke_data_clean.csv
```

---

### Notebook

`Stroke_Prediction.ipynb` contains the development workflow and experiments used during the project.

---

## 🛠️ Technologies

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit
- Plotly

### Model Persistence

- Joblib
- Pickle

### Development Tools

- Jupyter Notebook
- Visual Studio Code

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kagvoh/Stroke-Prediction-Project.git
```

Move into the project directory:

```bash
cd Stroke-Prediction-Project
```

---

### 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

---

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

The application requires both the FastAPI backend and Streamlit frontend to be running.

### 1. Start the Backend

From the project root:

```bash
uvicorn Backend.main:app --reload
```

The backend will normally run at:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

### 2. Start the Frontend

Open a second terminal and activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Then run:

```bash
streamlit run Frontend/app.py
```

The Streamlit application will normally be available at:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

---

## 🖥️ Application Features

### 📊 Stroke Rate Analytics

The application provides an analytics section for exploring observed stroke rates across different patient characteristics.

The analytics include:

- Age Group vs. Stroke Rate
- Hypertension vs. Stroke Rate
- Heart Disease vs. Stroke Rate
- BMI Category vs. Stroke Rate
- Glucose Category vs. Stroke Rate
- Smoking Status vs. Stroke Rate

The application also provides an overall insight highlighting the category with the highest observed stroke rate.

---

### 🧪 Stroke Risk Prediction

Users can enter patient information through the prediction interface.

The input information includes:

- Full name
- Age
- Gender
- Hypertension
- Heart disease
- BMI
- Average glucose level
- Smoking status
- Ever married
- Work type
- Residence type

Users can select one of the available machine learning models before making a prediction.

---

## 📋 Prediction Result

After submitting the patient information, the application displays:

- Patient name
- Patient age
- Date of prediction
- Probability of stroke
- Risk level
- Selected machine learning model

The predicted probability represents the model's estimated likelihood of stroke based on the provided patient information.

---

## 🔌 API

The FastAPI backend provides endpoints for communication with the Streamlit frontend.

### Prediction

```text
POST /predict
```

### Available Models

```text
GET /meta/models
```

### Input Options

```text
GET /meta/options
```

### Statistics

```text
GET /meta/statistics
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 🔄 Example Usage

```text
1. Start the FastAPI backend
              ↓
2. Start the Streamlit frontend
              ↓
3. Open the web application
              ↓
4. Enter patient information
              ↓
5. Select a machine learning model
              ↓
6. Submit the prediction request
              ↓
7. FastAPI processes the request
              ↓
8. Selected model generates a prediction
              ↓
9. Stroke probability and risk level are returned
              ↓
10. Result is displayed in the Streamlit interface
```

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The prediction generated by this application should **not be considered a medical diagnosis or a substitute for professional medical advice**.

Actual stroke risk assessment should be performed by qualified healthcare professionals using appropriate clinical information and medical examinations.

---

## 🔮 Future Improvements

Potential future improvements include:

- Improve the user interface and user experience.
- Add model explainability using SHAP or similar techniques.
- Add more clinically relevant features.
- Perform external validation using additional datasets.
- Add model monitoring.
- Deploy the application to a cloud platform.
- Improve prediction result visualization.

---

## 👨‍💻 Author

**Kagvoh**

GitHub: https://github.com/Kagvoh

---

## 📄 License

This project is intended for educational and research purposes.
