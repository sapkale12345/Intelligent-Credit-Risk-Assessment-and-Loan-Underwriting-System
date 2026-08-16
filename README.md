# Intelligent Credit Risk Assessment and Loan Underwriting System

An end-to-end **Machine Learning + RAG-based credit risk assessment and loan underwriting system** that predicts loan default risk, optimizes lending decisions using business costs, and generates evidence-based AI underwriting reports.

## Overview

Credit underwriting requires identifying borrowers who are likely to default while avoiding unnecessary rejection of creditworthy applicants.

This project develops a complete credit risk pipeline that:

* Performs exploratory analysis of borrower and loan characteristics
* Preprocesses numerical and categorical features
* Builds and compares multiple classification models
* Predicts the probability of loan default
* Optimizes the classification threshold using business cost
* Produces lending decisions based on predicted risk
* Uses a RAG pipeline to retrieve relevant historical credit-risk information
* Generates an AI underwriting report explaining the model's decision

The AI report **does not make a new lending decision**. It explains the machine-learning prediction using the retrieved Credit Risk Knowledge Guide.

## Problem Statement

Traditional credit assessment can involve evaluating multiple borrower attributes such as income, employment history, loan amount, interest rate, loan burden, loan grade, and previous default history.

The objective of this project is to build a data-driven underwriting system that can:

1. Estimate the probability of loan default.
2. Identify important risk patterns in borrower data.
3. Compare different machine-learning approaches.
4. Account for the asymmetric cost of lending errors.
5. Provide an interpretable underwriting report based on historical credit-risk analysis.

## Dataset

The dataset contains **32,581 loan applications and 13 variables** before cleaning.

Key features include:

* `person_age`
* `person_income`
* `person_home_ownership`
* `person_emp_length`
* `loan_intent`
* `loan_grade`
* `loan_amnt`
* `loan_int_rate`
* `loan_status`
* `loan_percent_income`
* `cb_person_default_on_file`
* `cb_person_cred_hist_length`

The target variable is `loan_status`, where:

* `0` = Non-Default
* `1` = Default

The original dataset contains approximately **21.87% default cases**.

## Data Preprocessing

The preprocessing workflow includes:

* Removed `application_id`
* Removed duplicate observations
* Handled missing numerical values using median imputation
* Handled categorical missing values using mode imputation
* Detected and capped numerical outliers using the IQR method
* Applied `StandardScaler` to numerical features
* Applied one-hot encoding to categorical features
* Used a stratified 80/20 train-test split

After removing duplicates, the dataset contained **32,414 observations**.

## Exploratory Data Analysis

The project analyzes how borrower and loan characteristics are associated with default risk.

### Key findings

* Default rate was substantially higher among lower-income borrowers.
* Very large loans showed a higher default rate than smaller loan categories.
* Higher interest-rate categories were associated with higher observed default rates.
* Very high loan-to-income burden showed a **46.73% observed default rate**.
* Loan grade showed a strong relationship with default risk, with default rates increasing substantially from Grade A through Grade G.
* Previous default history was associated with higher observed default rates.
* Home ownership and loan intent also showed differences in observed default rates.

For example, the observed default rate increased from **9.36% in the low interest-rate category to 44.72% in the very-high category**.

The very-high loan-percent-income burden category had an observed default rate of **46.73%**.

## Machine Learning Pipeline

### Models Evaluated

Three classification models were developed:

1. Logistic Regression
2. Random Forest
3. XGBoost

The preprocessing pipeline was integrated with the models using `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`. Logistic Regression additionally used SMOTEENN to address class imbalance.

## Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     80.61% |     53.87% |     79.06% |     64.08% |     87.19% |
| Random Forest       |     91.96% |     87.28% |     74.05% |     80.12% |     92.96% |
| **XGBoost**         | **92.81%** | **89.73%** | **75.81%** | **82.19%** | **94.28%** |

XGBoost achieved the strongest overall performance, with a **ROC AUC of 0.9428** and **F1 score of 0.8219** at the default classification threshold.

## Business Cost Analysis & Threshold Optimization

In credit lending, the cost of incorrectly approving a defaulter can be much higher than rejecting a good customer.

The project therefore assigns:

* False Positive cost = `1`
* False Negative cost = `11`

The XGBoost probability threshold was evaluated from **0.25 to 0.49** to identify the threshold that minimizes total business cost.

### Optimal Threshold

The minimum business cost was obtained at:

**Optimal threshold = 0.26**

**Minimum business cost = 3311**

At this threshold, the final classification results were:

* Accuracy: **91%**
* Precision: **78%**
* Recall: **81%**
* F1 Score: **80%**

This threshold-based approach prioritizes reducing costly false-negative lending decisions rather than relying only on the default probability threshold.

## AI Underwriting & RAG Pipeline

The project extends the ML model with a retrieval-augmented generation pipeline.

The workflow:

```text
Applicant Information
        ↓
Feature Processing
        ↓
XGBoost Model
        ↓
Probability of Default
        ↓
Risk Score & Lending Decision
        ↓
Retrieve Relevant Credit-Risk Context
        ↓
RAG-based AI Underwriting Report
```

A **Credit Risk Knowledge Guide** is loaded, split into chunks, embedded using:

`sentence-transformers/all-MiniLM-L6-v2`

and stored in a **FAISS vector database**. The retriever uses MMR-based retrieval to obtain relevant context.

The generated report covers:

* Risk Summary
* Feature-wise Explanation
* Decision Justification

The prompt explicitly restricts the AI to the retrieved Credit Risk Knowledge Guide and prevents it from making a new lending decision.

## Risk Decision Framework

The system converts the predicted probability of default into a risk score:

```text
Risk Score = Probability of Default × 100
```

The decision framework is:

| Risk Score | Decision      |
| ---------: | ------------- |
|     `< 20` | APPROVE       |
| `20 – <50` | MANUAL REVIEW |
|     `≥ 50` | REJECT        |

These rules are implemented after obtaining the model's predicted probability.

## Example Output

For a sample applicant, the system produced:

* Probability of Default: **0.0112**
* Risk Score: **1.12**
* Decision: **APPROVE**

The RAG component then generated a feature-wise underwriting explanation based on the retrieved historical credit-risk information.

## Technologies Used

**Programming:**
Python

**Data Analysis:**
Pandas | NumPy

**Visualization:**
Matplotlib | Seaborn

**Machine Learning:**
Scikit-learn | Imbalanced-learn | XGBoost

**NLP / Generative AI:**
LangChain | LangChain-Groq | Sentence Transformers

**Vector Database:**
FAISS

**Development Environment:**
Google Colab

## Project Workflow

```text
Data Loading
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Missing Value Treatment
     ↓
Outlier Treatment
     ↓
Exploratory Data Analysis
     ↓
Feature & Target Split
     ↓
Train-Test Split
     ↓
Preprocessing Pipeline
     ↓
Model Development
     ↓
Model Comparison
     ↓
XGBoost Selection
     ↓
Business Cost Analysis
     ↓
Threshold Optimization
     ↓
Final Risk Prediction
     ↓
RAG-based Underwriting Explanation
```

## Repository Structure

```text
Credit-Risk-Assessment-and-Loan-Underwriting/
│
├── README.md
└── Intelligent_Credit_Risk_Assessment_and_Loan_Underwriting_System.ipynb
```

## How to Run

1. Open the notebook in Google Colab.
2. Upload/provide the required credit-risk dataset.
3. Install the required Python packages.
4. Run the notebook sequentially.
5. For the RAG underwriting component, provide the required Credit Risk Knowledge Guide PDF.
6. Configure the required API credentials securely using environment variables or Colab Secrets.

**Never place API keys directly inside the notebook or GitHub repository.**

## Key Takeaways

* XGBoost achieved the best overall classification performance among the evaluated models.
* Default risk showed strong variation across income, loan amount, interest rate, loan burden, loan grade, and previous default history.
* Business-cost optimization provided a more decision-oriented threshold than relying only on the conventional 0.50 cutoff.
* The RAG component adds contextual underwriting explanations to the model output while keeping the lending decision controlled by the machine-learning model.

## Disclaimer

This project is developed for **educational and analytical purposes**. The model and underwriting framework should not be used as a standalone system for real-world lending decisions without appropriate validation, regulatory review, fairness assessment, monitoring, and domain expertise.
