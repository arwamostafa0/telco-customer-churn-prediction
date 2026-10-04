# 📊 Telco Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to churn based on their demographic information, services, contract, and billing details.

## 🚀 Project Overview

Customer churn is an important business problem for telecom companies. The goal of this project is to build a machine learning model that can identify customers who are at higher risk of leaving the service.

The trained model is integrated into an interactive Streamlit application where users can enter customer information and receive a churn prediction with its estimated probability.

## 🎯 Objective

The main objectives of this project are:

* Explore and understand customer churn patterns.
* Clean and preprocess the dataset.
* Handle categorical and numerical features.
* Compare multiple machine learning models.
* Evaluate models using metrics beyond accuracy.
* Select a decision threshold based on validation F1-score.
* Build an interactive Streamlit application for prediction.

## 🧠 Machine Learning Workflow

The project follows these main steps:

1. Data cleaning
2. Exploratory Data Analysis
3. Feature preprocessing
4. Train/test split
5. Model training
6. Model comparison
7. Hyperparameter tuning
8. Threshold optimization
9. Final model training
10. Streamlit deployment

## 🤖 Models Evaluated

The following models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

Logistic Regression was selected as the final model because it provided a strong balance between the evaluation metrics used for this project.

## 📈 Final Model Performance

The final Logistic Regression model was evaluated on the test set using a decision threshold of **0.40**.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 80.20% |
| Precision | 61.18% |
| Recall    | 69.52% |
| F1-score  | 65.08% |

The 0.40 threshold was selected using validation data to improve the balance between precision and recall, with F1-score used as the selection metric.

## 🖥️ Streamlit Application

The application allows users to enter:

* Customer demographics
* Tenure
* Phone and internet services
* Additional services
* Contract information
* Billing information
* Payment method

The application then returns:

* Churn probability
* Churn risk prediction
* Customer profile summary

## 🛠️ Technologies

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Jupyter / Google Colab
* PyCharm

## 📁 Project Structure

```text
telco-churn-prediction/
│
├── app.py
├── telco_churn_prediction.ipynb
├── Telco-Customer-Churn.csv
├── final_model.pkl
├── preprocessor.pkl
├── requirements.txt
└── README.md

```

## ▶️ Run Locally

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

The application will open locally in your browser.

## ⚠️ Limitations

This project is a machine learning prototype and is not intended to be used as a production decision-making system without further validation.

The model's predictions represent estimated churn risk based on the available dataset and should not be interpreted as causal conclusions.

Further improvements could include:

* More extensive feature engineering
* More robust cross-validation pipelines
* Additional boosting models
* Model explainability techniques
* More extensive error analysis
* Monitoring after deployment
