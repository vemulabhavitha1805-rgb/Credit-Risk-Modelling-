# 💳 IntelliRisk | AI-Powered Credit Risk Predictive Platform

![IntelliRisk Screenshot](file:///Users/bhavitha/.gemini/antigravity/brain/218f2674-fab7-4025-9c16-63f7e25b60ed/.system_generated/click_feedback/click_feedback_1779026452396.png)

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
└── README.md                         # Project Documentation
```

---

## 🚀 How to Run the Web Application

The application is built on top of **Streamlit** and requires a Python 3 environment.

### Prerequisites

Ensure you have Python installed, along with the required libraries. If you are using the pre-configured virtual environment:

1. **Open your Terminal** (on macOS).
2. **Navigate to the Project Folder**:
   ```bash
   cd "/Users/bhavitha/Documents/Projects /Credit risk modelling /Credit-Risk-Modelling-"
   ```
3. **Activate the Virtual Environment**:
   ```bash
   source ../.venv/bin/activate
   ```
4. **Launch the Server**:
   ```bash
   streamlit run app.py
   ```
5. **Open in Browser**: The terminal will print a local URL. Navigate to **[http://localhost:8501](http://localhost:8501)** to interact with your model!

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

---

## 🎨 Premium UI Aesthetics Details

The application overrides default Streamlit UI styling via scoped HTML/CSS injections:
* **Font Styling**: Loaded Google Font `Outfit` for modern, geometric look and feel.
* **Palette**: Curated deep-gray backgrounds (`#0d1117`), sleek blue gradient highlights, and soft green/red glowing border shadows representing final classification states.
* **Hover Transitions**: Premium tactile feedback when hovering over sliders, selection cards, and button elements.

---

## 📄 License
This project utilizes the public-domain German Credit Dataset. Feel free to copy, modify, and build on top of this repository. Developed as a credit risk analytics demonstration platform.