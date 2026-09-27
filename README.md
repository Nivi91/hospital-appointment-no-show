# Hospital Appointment No-Show Prediction

## Project Overview

This project predicts the risk of a patient missing a scheduled hospital appointment.

The goal is to help hospital staff identify appointments that may have a higher risk of no-show so that appropriate follow-up actions, such as reminders or appointment confirmation, can be considered.

The project includes data preprocessing, feature engineering, patient appointment history, machine learning model development, cost-sensitive learning, threshold selection, model evaluation, prediction explanation, and a Streamlit application.

---

## Dataset

The project uses a hospital appointment dataset containing patient, appointment, medical, reminder, and appointment outcome information.

After data cleaning, the modelling dataset contains:

- 110,522 appointments
- 20 modelling features
- 88,208 attended appointments
- 22,314 no-show appointments

The target variable is:

- `0` – Appointment attended
- `1` – Appointment no-show

---

## Data Preparation

The data preparation process includes:

- Converting appointment and scheduling dates into datetime format
- Calculating waiting days between scheduling and appointment dates
- Removing records with invalid negative waiting days
- Replacing invalid age values with the median age
- Creating appointment weekday, month, day, and scheduled-hour features
- Creating an appointment weekend indicator
- Creating age groups
- Creating waiting-time groups
- Creating previous appointment history features

Patient history features include:

- Previous appointments
- Previous no-shows
- Previous no-show rate

The patient history features are calculated using previous appointments only so that information from the current appointment is not used as historical information.

---

## Features

The model uses 20 features:

- Gender
- Age
- Neighbourhood
- Scholarship
- Hypertension
- Diabetes
- Alcoholism
- Handicap
- SMS received
- Waiting days
- Appointment weekday
- Appointment month
- Appointment day
- Scheduled hour
- Age group
- Appointment weekend
- Waiting-time group
- Previous appointments
- Previous no-shows
- Previous no-show rate

The preprocessing pipeline transforms these features into 115 processed features using numerical scaling and categorical one-hot encoding.

---

## Train, Validation and Test Strategy

A temporal split was used instead of a random split.

This keeps earlier appointments for training and later appointments for validation and testing.

The final test set contains 21,987 appointments and is used as an unseen dataset for final model evaluation.

---

## Machine Learning Approach

The project compares different machine learning approaches, including:

- Logistic Regression baseline
- Cost-sensitive Logistic Regression
- Random Forest
- Different class-weight cost ratios
- Threshold optimization

Because failing to identify a patient who may miss an appointment can be more costly than generating an additional alert, cost-sensitive learning was used to give greater importance to the no-show class.

---

## Cost-Sensitive Model Comparison

The validation results for the tested cost ratios are:

| Cost Ratio | Best Threshold | Validation Cost | No-show Recall |
|------------|----------------|-----------------|----------------|
| 2:1 | 0.56 | 4801 | 18.61% |
| 3:1 | 0.52 | 6285 | 63.64% |
| 4:1 | 0.51 | 7003 | 81.23% |
| 5:1 | 0.45 | 7287 | 91.57% |

The final model uses the 4:1 cost-sensitive Logistic Regression model.

A threshold of 0.50 was selected as the final operational threshold to keep the decision boundary consistent with the Streamlit risk categories. The lowest validation cost for the 4:1 model occurred at 0.51.

---

## Final Model

The final model is:

**Cost-sensitive Logistic Regression with a 4:1 class-weight ratio**

The final operational threshold is:

**0.50**

The prediction is classified into three risk categories:

- **Below 30%** → Low Risk
- **30% to below 50%** → Medium Risk
- **50% or above** → High Risk

---

## Final Test Performance

The final model was evaluated on the unseen test set.

Results:

- Accuracy: **56%**
- ROC-AUC: **0.721**
- PR-AUC: **0.324**
- No-show recall: **83%**

Confusion matrix:


[[9024, 8893],
 [ 701, 3369]]

 
```text
## Model Explainability


The Streamlit application provides an explanation of the factors influencing each prediction.

For Logistic Regression, feature contributions are calculated from the processed feature values and model coefficients.

The application separates the displayed factors into:

Factors increasing no-show risk
Factors reducing no-show risk

These factors represent model associations and should not be interpreted as direct causes of a patient missing an appointment.

## Streamlit Application

The Streamlit application provides a receptionist-friendly interface.

The user can:

Enter a Patient ID
Select an appointment
View patient details
View appointment details
Automatically view previous appointment history
Calculate the no-show risk score
View the risk category
See a recommended action
View the key factors influencing the prediction

## Recommended Actions

High Risk

Contact the patient and confirm the appointment.

Medium Risk

Consider sending a reminder to the patient.

Low Risk

No additional action is required.

## Project Files
no_show_app/
│
├── 01_no_show_prediction.ipynb
├── app.py
├── hospital_appointment.csv
├── no_show_model.pkl
├── no_show_preprocessor.pkl
├── requirements.txt
└── README.md

## File Description

01_no_show_prediction.ipynb – Complete data analysis, feature engineering, model development, evaluation, threshold analysis, and explainability
app.py – Streamlit application
hospital_appointment.csv – Hospital appointment dataset
no_show_model.pkl – Saved final machine learning model
no_show_preprocessor.pkl – Saved preprocessing pipeline
requirements.txt – Required Python packages
README.md – Project documentation

## How to Run

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

The application will open in the browser.

## Important Note

The displayed value is an estimated no-show risk score produced by the machine learning model.

The prediction is intended as a decision-support tool for appointment follow-up and should not be treated as a definitive statement about whether a patient will miss an appointment.