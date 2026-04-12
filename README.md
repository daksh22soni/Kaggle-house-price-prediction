# 🏠 Kaggle House Price Prediction

> A production-grade, stacked ensemble machine learning system for predicting residential property sale prices — integrated with a full DevOps pipeline including containerisation, CI/CD automation, and real-time monitoring.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker)](https://www.docker.com/)
[![Jenkins](https://img.shields.io/badge/CI%2FCD-Jenkins-D33833?logo=jenkins)](https://www.jenkins.io/)
[![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-E6522C?logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Dashboard-Grafana-F46800?logo=grafana)](https://grafana.com/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Project Architecture](#2-project-architecture)
3. [Features](#3-features)
4. [Installation Guide](#4-installation-guide)
5. [How to Run the Project](#5-how-to-run-the-project)
6. [Output / Results](#6-output--results)
7. [Folder Structure](#7-folder-structure)
8. [Important Notes](#8-important-notes)
9. [Future Scope](#9-future-scope)
10. [Authors / Credits](#10-authors--credits)

---

## 1. Overview

### Abstract

This project presents an end-to-end machine learning application for predicting residential house sale prices using the **Kaggle Advanced Regression Techniques dataset**. The core predictive system employs a **stacked ensemble model** combining ElasticNet, XGBoost, and CatBoost as base learners, with a meta-model for final price estimation. Beyond the modelling component, this project implements a complete **DevOps and Agile engineering pipeline** — treating the ML model as a production service rather than an isolated notebook experiment.

### Problem Statement

Accurate real estate price estimation is a high-value, data-rich problem influenced by dozens of structural, locational, and temporal features. Simple regression models fail to capture non-linear feature interactions and heteroscedastic variance in property pricing. The challenge is to build a model that is not only accurate, but also **reproducible**, **deployable**, and **observable** in a production-like environment.

### Objectives

- Build a stacked ensemble regression model capable of accurately predicting house sale prices from structured feature data.
- Develop a real-time **Streamlit web application** to serve model predictions interactively.
- Containerise the application using **Docker** for environment reproducibility.
- Automate builds and deployments through a **Jenkins CI/CD pipeline**.
- Instrument the application with **Prometheus metrics** and visualise system behaviour through a **Grafana dashboard**.
- Apply **Agile practices** (iterative commits, fast-fail pipelines, short feedback loops) throughout the development lifecycle.

---

## 2. Project Architecture

### System Workflow

```
User Input (Streamlit UI)
        │
        ▼
Feature Preprocessing (src/)
        │
        ▼
┌───────────────────────────────┐
│     Stacked Ensemble Model    │
│  ┌──────────┐  ┌───────────┐  │
│  │ElasticNet│  │  XGBoost  │  │
│  └──────────┘  └───────────┘  │
│        ┌──────────────┐       │
│        │   CatBoost   │       │
│        └──────────────┘       │
│              │                │
│       Meta-Model (Stacker)    │
└───────────────────────────────┘
        │
        ▼
Predicted Sale Price (USD)
        │
        ▼
Prometheus /metrics endpoint (port 8000)
        │
        ▼
Grafana Dashboard (port 3000)
```

### DevOps Pipeline Overview

```
git push → GitHub → Jenkins (Webhook)
                        │
              ┌─────────▼──────────┐
              │  Clone Repository  │
              │  Build Docker Image│
              │  Stop Old Container│
              │  Run New Container │
              │  Health Check      │
              └────────────────────┘
                        │
              Docker Container (port 8501)
                        │
              Prometheus Scrape (port 8000)
                        │
              Grafana Dashboard (port 3000)
```

### Modules and Components

| Component | Technology | Description |
|---|---|---|
| **Web Application** | Streamlit | Interactive UI for user inputs and displaying predictions |
| **Preprocessing** | Scikit-learn, Pandas | Feature engineering, encoding, and normalisation pipeline |
| **Base Models** | ElasticNet, XGBoost, CatBoost | First-level ensemble learners |
| **Meta-Model** | Stacking Regressor | Second-level learner combining base model outputs |
| **Containerisation** | Docker | Single reproducible image; Dockerfile at project root |
| **CI/CD Pipeline** | Jenkins (Declarative) | 6-stage automated build, deploy, and health-check pipeline |
| **Metrics Collection** | Prometheus Client | Custom `/metrics` endpoint with 10 instrumented metrics |
| **Visualisation** | Grafana | Two-section dashboard: Data Insights & Model Performance |

### Models and Rationale

- **ElasticNet** — Linear model with L1+L2 regularisation. Captures linear relationships and handles multicollinearity in correlated housing features (e.g., GrLivArea vs. TotRmsAbvGrd).
- **XGBoost** — Gradient boosted trees. Handles non-linear interactions and missing values natively; strong performance on tabular regression benchmarks.
- **CatBoost** — Gradient boosting with native categorical feature support. Efficient and robust on mixed-type datasets without extensive encoding.
- **Stacking Meta-Model** — Combines out-of-fold predictions from all three base learners to reduce individual model bias and improve generalisation.

---

## 3. Features

- **Stacked Ensemble Regression** combining ElasticNet, XGBoost, and CatBoost with a meta-learner for superior prediction accuracy.
- **Real-time Streamlit Web App** enabling interactive house price prediction from user-provided feature inputs.
- **Dockerised Deployment** ensuring fully reproducible, environment-agnostic application builds.
- **Jenkins CI/CD Pipeline** with 6 automated stages: clone, build, stop, deploy, and health check — triggered on every `git push`.
- **Prometheus Instrumentation** with 10 custom application metrics covering request counts, prediction latency, model-level latency, model disagreement, error rates, and input feature distributions.
- **Grafana Dashboard** with two logical sections — *Data Insights* (input trends, feature distributions, price forecasts) and *Model Performance* (latency, drift, error rate, service health).
- **Agile Development Workflow** with 49 iterative commits, fast-fail pipeline stages, and near-real-time monitoring feedback loops.
- **Clean Repository Structure** with `.gitignore` and `.dockerignore` files to maintain lean version history and efficient Docker builds.

---

## 4. Installation Guide

### Prerequisites

> ⚠️ **This project requires Python 3.10 or Python 3.11 exclusively.** Other versions are not supported and may cause dependency or runtime errors.

Download the correct Python version from the official source:

- 🐍 **Python 3.10**: [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)
- 🐍 **Python 3.11**: [https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/)

Verify your Python version after installation:

```bash
python --version
# Expected: Python 3.10.x  or  Python 3.11.x
```

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/daksh22soni/Kaggle-house-price-prediction.git
cd Kaggle-House-Price-Prediction
```

---

### Step 2 — Set Up a Virtual Environment *(Recommended)*

Using a virtual environment isolates project dependencies from your system Python installation.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` prefixed in your terminal prompt, confirming activation.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The dependency stack includes large ML packages (XGBoost, CatBoost, PyTorch-related libraries). Installation may take a few minutes depending on your network speed.

---

### Step 4 — (Optional) Docker Setup

If you prefer to run the application in a Docker container (recommended for production-like deployment):

```bash
# Build the Docker image
docker build -t house-price-app .

# Run the container
docker run -d -p 8501:8501 --name house-price-container house-price-app
```

---

## 5. How to Run the Project

### Running Locally (Streamlit App)

After completing the installation steps, launch the Streamlit web application:

```bash
streamlit run app/app.py
```

The application will be accessible in your browser at:

```
http://localhost:8501
```

### Running via Docker

```bash
docker run -d -p 8501:8501 --name house-price-container house-price-app
```

Access the application at `http://localhost:8501`.

To view Prometheus metrics:
```
http://localhost:8000/metrics
```

### Running the Jenkins Pipeline

Ensure Jenkins is configured with the repository URL in the Jenkinsfile. The pipeline will automatically trigger on a `git push` to the `main` branch and execute the following stages in sequence:

```
1. Clone Repository      → Pull latest code from GitHub
2. Build Docker Image    → docker build -t house-price-app .
3. Stop Old Container    → docker stop / docker rm (safe fail)
4. Run New Container     → docker run -d -p 8501:8501
5. Health Check          → docker ps (verify container is active)
```

### Accessing Grafana Dashboard

Once the monitoring stack is running, open Grafana at:

```
http://localhost:3000
```

Connect Prometheus as a data source at `http://localhost:8000/metrics` and import the *House Price App* dashboard.

---

## 6. Output / Results

### Application Output

Upon submitting house features through the Streamlit UI, the application returns:

- **Predicted Sale Price** in USD (e.g., `$125,000 – $150,000` for typical Ames, Iowa properties).
- Prediction is computed by the stacked ensemble in real time and displayed on the same page.

### Prometheus Metrics (Sample)

| Metric | Type | Description |
|---|---|---|
| `app_requests_total` | Counter | Total prediction requests received |
| `prediction_latency_seconds` | Summary | End-to-end latency per prediction |
| `app_errors_total` | Counter | Total prediction errors |
| `model_latency_seconds` | Summary | Per-model latency (ElasticNet, XGBoost, CatBoost, Meta) |
| `model_disagreement` | Summary | Std. deviation across base model outputs |
| `prediction_values` | Histogram | Distribution of final predicted prices |
| `input_feature_distribution` | Histogram | Distribution of key input features |

### Observed Performance

- **Error Rate**: 0% across all recorded prediction requests.
- **End-to-End Prediction Latency**: ~939ms (single request, Docker container).
- **Per-Model Latency Breakdown**:
  - ElasticNet: < 2ms
  - CatBoost: ~28ms
  - XGBoost: ~260ms (highest — expected for tree-based inference)
  - Meta-Model: < 2ms
- **Jenkins Pipeline Duration**: ~9–15 minutes (dominated by Docker image build — dependency installation from `requirements.txt`).
- **Grafana Dashboard Refresh Rate**: Every 5 seconds.

---

## 7. Folder Structure

```
Kaggle-House-Price-Prediction/
│
├── app/                        # Streamlit web application source code
│   └── app.py                  # Main application entry point
│
├── src/                        # Core ML logic
│   ├── preprocessing.py        # Feature engineering and encoding pipeline
│   └── predict.py              # Prediction inference logic
│
├── models/                     # Serialised trained model files (.pkl / .joblib)
│
├── notebooks/                  # Jupyter notebooks for EDA and experimentation
│
├── Submissions/                # Kaggle competition submission CSV files
│
├── Dockerfile                  # Docker image definition (Python 3.10 base)
├── .dockerignore               # Files excluded from Docker build context
├── Jenkinsfile                 # Declarative Jenkins CI/CD pipeline definition
├── requirements.txt            # Python dependency list
├── .gitignore                  # Files excluded from version control
└── README.md                   # Project documentation (this file)
```

---

## 8. Important Notes

> ### ⚠️ Python Version Compatibility Warning
>
> **This project is strictly compatible with Python 3.10 or 3.11.**
> Using Python 3.8, 3.9, 3.12, or any other version may cause dependency resolution failures, incompatible package builds, or silent runtime errors.
>
> Please verify your Python version with `python --version` before proceeding with installation.

Additional notes:

- **Large model files** (`.pkl`, `.joblib`) and **raw data files** are excluded from the repository via `.gitignore`. Ensure trained model files are present in the `models/` directory before running inference.
- **Port availability**: Ensure ports `8501` (Streamlit), `8000` (Prometheus), and `3000` (Grafana) are free before starting the application stack.
- **Docker prerequisite**: Docker must be installed and the Docker daemon must be running for both the containerised application and the Jenkins pipeline to function.
- The Jenkins pipeline uses `|| true` in the *Stop Old Container* stage to prevent pipeline failure when no existing container is found (e.g., on the first run).

---

## 9. Future Scope

- **Automated Hyperparameter Tuning**: Integrate Optuna or Bayesian optimisation into the training pipeline for continuous model improvement beyond the current ensemble configuration.
- **Model Registry Integration**: Incorporate MLflow or DVC for model versioning, experiment tracking, and reproducible training runs.
- **Kubernetes Orchestration**: Migrate from single-container Docker deployment to a Kubernetes cluster for horizontal scaling, rolling updates, and high availability.
- **Data Drift Detection**: Add statistical drift detection (e.g., using Evidently AI or Alibi Detect) to the Prometheus metrics stack, enabling automated alerts when input distributions deviate from training data.
- **Automated Retraining Triggers**: Implement a feedback loop where accumulated prediction data and ground-truth labels trigger scheduled model retraining via the Jenkins pipeline.
- **Extended Feature Engineering**: Explore geospatial features (neighbourhood proximity, school district ratings) and temporal market trends to improve model generalisation beyond the Ames, Iowa dataset.
- **API Layer**: Expose the model as a RESTful API (FastAPI / Flask) in addition to the Streamlit interface, enabling integration with third-party applications.
- **Formal Agile Sprint Structure**: Adopt a structured Scrum board (Jira or GitHub Projects) for sprint planning, backlog management, and velocity tracking in future iterations.

---

## 10. Authors / Credits

### Team Members

| Name | SAP ID | Role |
|---|---|---|
| **Aditya Madhav Mantri** | 70562300021 | ML Modelling, Repository Management, CI/CD |
| **Daksh Soni** | 70562300080 | DevOps Integration, Prometheus Instrumentation |
| **Priyanshu Nayak** | 70562300030 | Docker Containerisation, Streamlit Development |
| **Simar Singh Khanuja** | 70562200088 | Grafana Dashboarding, Agile Documentation |

### Academic Details

| Field | Details |
|---|---|
| **Institution** | SVKM's NMIMS (Deemed-to-be-University), Indore Campus |
| **School** | School of Technology Management & Engineering |
| **Programme** | B.Tech Artificial Intelligence & Data Science |
| **Semester** | Semester 6 |
| **Academic Year** | 2025–26 |
| **Faculty Supervisor** | Dr. Gaurav Paliwal |

### Dataset

- **Kaggle House Prices: Advanced Regression Techniques**
  - [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

### Tools & Frameworks

- [Streamlit](https://streamlit.io/) · [Scikit-learn](https://scikit-learn.org/) · [XGBoost](https://xgboost.readthedocs.io/) · [CatBoost](https://catboost.ai/)
- [Docker](https://www.docker.com/) · [Jenkins](https://www.jenkins.io/) · [Prometheus](https://prometheus.io/) · [Grafana](https://grafana.com/)
- [Git](https://git-scm.com/) · [GitHub](https://github.com/)

---

<div align="center">

*Developed as part of the DevOps & Agile Practices curriculum — NMIMS Indore, 2025–26*

</div>
