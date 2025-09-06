# clinicalPrediction Project
This project is an end-to-end clinical prediction system for depression detection using PHQ-9 survey data and lifestyle inputs. It combines a React frontend, a Spring Boot backend and a Flask-based machine learning microservice.

---

## Project Overview

- Goal: Predict likelihood of depression from PHQ-9 responses using multiple machine learning models and explain the prediction with SHAP values.
- Models Used: Logistic Regression, Random Forest, XGBoost
- Explainability: SHAP values displayed visually in the UI.
- User Feedback Loop: Users can flag incorrect predictions to improve the system over time.

---

## Technology Stack

| Layer      | Tech                 |
|------------|----------------------|
| Frontend   | React.js + Bootstrap |
| Backend    | Java Spring Boot     |
| ML Service | Python Flask         |
| ML Models  | Scikit-learn, XGBoost, SHAP |
| Data Store | PostgreSQL (via Spring Data JPA) |

---

## Features

- Dynamic form for PHQ-9 and lifestyle inputs
- Multiple model selection with performance metrics (accuracy, F1, ROC AUC)
- SHAP visualizations of top contributing features
- User feedback form to flag incorrect predictions
- Seamless integration between frontend, backend, and ML service

---

## Folder Structure
ClinicalPrediction/
├── backend/ # Spring Boot backend (Java)
├── frontend/ # React frontend (JS)
├── ml-service/ # Flask ML service (Python)

---

## How to Run the Project

### 1. Start the ML Microservice (Flask)

cd ml-service
pip install -r requirements.txt
python app.py

## This runs on http://127.0.0.1:8000

### 2. Start the Backend (Sprint Boot)

cd backend
./mvnw spring-boot:run

## This runs on http://localhost:8080

### 3. Start the Frontend (React)

cd frontend
npm install
npm start

## This opens at http://localhost:3000

---

## Model Metrics Example:

After a prediction, metrics like the following are shown:

Accuracy: 0.85

Precision: 0.82

Recall: 0.87

F1 Score: 0.84

ROC AUC: 0.91

Cross-validation performance is also displayed with mean ± std deviation.

## SHAP Visualizations

The frontend renders a horizontal bar chart of top SHAP features, helping the user understand what influenced the model’s decision.

## Dataset

The system is trained on a PHQ-9 dataset collected from university students. Due to limitations in data diversity (age, occupation), future improvements will aim to generalize beyond student samples.

## Feedback Flow

Users can submit free-text feedback after each prediction.

The system stores this feedback for further analysis and potential model improvements.

---

Author:

Name: Jair Josue Ortega Pachito

MSc Computer Science (Birkbeck, University of London)

GitHub: https://github.com/JairJOP/clinicalPrediction