# 💳 IntelliRisk | AI-Powered Credit Risk Predictive Platform

An end-to-end Machine Learning web application designed to evaluate credit applicant profiles and predict credit default risks ("Good" vs "Bad" risk). Built with **Streamlit** for a premium glassmorphic dark UI and backed by a robust **Random Forest Classification** model trained on the standard **German Credit Dataset**.

---

## 🌟 Key Features

* **Premium Glassmorphic Interface**: Sleek dark-mode dashboard with custom typography (`Outfit` from Google Fonts), smooth CSS transitions, interactive slide controls, and visual feedback states.
* **Real-Time ML Predictions**: Instantly analyzes credit requests and displays either a **Low Risk** (Approved) or **High Risk** (Rejected/Secured) result panel accompanied by a dynamic prediction confidence score.
* **Comprehensive Profile Evaluation**: Inputs details across both applicant demographics and credit metrics:
  * **Demographics**: Age, Sex, Job Category (Unskilled to Management), and Housing Status (Own, Rent, Free).
  * **Financial Metrics**: Requested Credit Amount (€), Loan Duration (Months), Savings Balance, Checking Account Balance, and Loan Purpose (e.g., Car, Education, Business, Furniture).
* **Robust Pre-Processing & Validation**: Cached model load pipeline with built-in exception handling to ensure rapid, fail-safe inference.

---

## 📂 Repository Architecture

```
Credit-Risk-Modelling-/
├── german_credit_data.csv            # Original German Credit Dataset (1,000 observations)
├── analysis_model.ipynb              # Jupyter Notebook for EDA, Preprocessing, and Model Training
├── app.py                            # Streamlit Production Dashboard Source Code
├── random_trees_credit_model.pkl    # Serialized Random Forest Classifier
├── target_encoder.pkl                # Serialized Target Encoder
├── Sex_label_encoder.pkl             # Serialized Categorical Label Encoders
├── Housing_label_encoder.pkl
├── Saving accounts_label_encoder.pkl
├── Checking account_label_encoder.pkl
├── Purpose_label_encoder.pkl
├── requirements.txt                  # Runtime dependencies
├── tests/test_app.py                  # Dashboard smoke tests
├── .gitignore                        # Local/cache file exclusions
└── README.md                         # Project Documentation
```

---

## 🚀 How to Run the Web Application

The application is built on top of **Streamlit** and requires a Python 3 environment.

### Prerequisites

Ensure you have Python 3.10 or newer installed. From a terminal:

1. **Navigate to the project folder**:
   ```bash
   cd Credit-Risk-Modelling-
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```
   On macOS, install the OpenMP runtime before running the notebook's XGBoost
   comparison:
   ```bash
   brew install libomp
   ```
4. **Launch the server**:
   ```bash
   python -m streamlit run app.py
   ```
5. **Open in Browser**: The terminal will print a local URL. Navigate to **[http://localhost:8501](http://localhost:8501)** to interact with your model!

### Verify the project

Run the dashboard smoke tests after installing the dependencies:

```bash
python -m unittest discover -s tests -v
```

The checks load the dashboard and run an assessment with both account fields set
to **No Account**, which confirms the saved encoders and missing-value handling
remain compatible.

---

## 🧠 Machine Learning Pipeline

### 1. Data Cleaning & Exploratory Analysis
* **Data Source**: The **German Credit Dataset** consisting of 1,000 credit applicant profiles with 10 features (numerical and categorical) and a binary risk target (`good` vs `bad`).
* **Missing Value Imputation**: Handled missing categorical values in `Saving accounts` and `Checking account` by treating them as independent "No Account" categories (or `NaN` mapping) to preserve predictive signals from unbanked applicants.

### 2. Feature Engineering & Pre-Processing
* Features encoded using dedicated `LabelEncoder` objects serialized individually to allow stable out-of-sample predictions during inference.
* Features incorporated in the model:
  * **Numerical**: `Age`, `Credit amount`, `Duration`
  * **Categorical**: `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Purpose`

### 3. Model Training & Serialization
* Evaluated standard classification architectures (including Decision Trees, Random Forests, and XGBoost).
* Fine-tuned a high-performing **Random Forest Classifier** via cross-validated Grid Search (`GridSearchCV`) to optimize accuracy and model robustness.
* Serialized all components (`.pkl`) to guarantee zero feature-mismatch during runtime.

### Responsible use

IntelliRisk is an educational demonstration built from a small historical
dataset. Its output is a model estimate, not a lending decision. Do not use it
as the sole basis for approving, rejecting, pricing, or otherwise determining
access to credit; a real deployment needs appropriate governance, validation,
fairness review, monitoring, and human oversight.

---

## 🎨 Premium UI Aesthetics Details

The application overrides default Streamlit UI styling via scoped HTML/CSS injections:
* **Font Styling**: Loaded Google Font `Outfit` for modern, geometric look and feel.
* **Palette**: Curated deep-gray backgrounds (`#0d1117`), sleek blue gradient highlights, and soft green/red glowing border shadows representing final classification states.
* **Hover Transitions**: Premium tactile feedback when hovering over sliders, selection cards, and button elements.

---

## 📄 License
This project utilizes the public-domain German Credit Dataset. Feel free to copy, modify, and build on top of this repository. Developed as a credit risk analytics demonstration platform.
