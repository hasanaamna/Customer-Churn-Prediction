# 🏠 Customer Churn Prediction

An end-to-end machine learning application that predicts whether a customer is likely to churn based on their demographic information, account details, services, contract, and billing information.

The project combines **machine learning, data analysis, FastAPI, and an interactive web dashboard** into a complete customer churn prediction system.

---

## 🚀 Project Overview

Customer churn is a major challenge for businesses because retaining existing customers is often more valuable than acquiring new ones.

This project uses the **IBM Telco Customer Churn dataset** to build a machine learning system that:

- Analyzes customer churn patterns
- Preprocesses customer data
- Trains multiple machine learning models
- Compares model performance
- Tunes the classification threshold
- Predicts customer churn probability
- Provides an interactive FastAPI backend
- Provides a modern analytics dashboard
- Displays customer risk information and model insights

---

## ✨ Features

### Machine Learning

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Logistic Regression
- Random Forest
- Model comparison
- Class imbalance handling
- Probability-based churn prediction
- Classification threshold tuning
- ROC-AUC evaluation
- Precision, Recall and F1-score evaluation

### Backend

- FastAPI REST API
- `/health` endpoint
- `/predict` endpoint
- Pydantic request validation
- CORS support
- Saved machine learning model
- Configurable prediction threshold

### Frontend

- Interactive customer analysis dashboard
- Churn risk visualization
- Customer profile section
- Customer services overview
- Risk indicators
- Model performance metrics
- Responsive dark-themed UI

---

## 🧠 Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Train / Test Split
       ↓
Preprocessing Pipeline
       ↓
Model Training
       ↓
Model Comparison
       ↓
Threshold Tuning
       ↓
Final Model
       ↓
FastAPI Prediction API
       ↓
Interactive Dashboard