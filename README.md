# ChurnGuard AI 🛡️
> **Customer Churn Prediction powered by Machine Learning**

An interactive Streamlit web application that predicts customer churn probability in telecommunications, evaluated across 4 machine learning models on the IBM Telco dataset.

---

## 📋 Features

- **Light & Professional UI:** Clean white card architecture, soft shadows, and indigo accents.
- **Multi-Model Selector:** Switch in real-time between **Logistic Regression, Decision Tree, Random Forest, and KNN**.
- **Interactive Threshold Tuning:** Dynamic slider initialized to each model's calibrated F1-optimum ($\ge 0.45$).
- **Risk Assessment & Confidence Meter:** Real-time probability gauge with color-coded risk badges.
- **Actionable Retention Insights:** Automatically scans customer profiles and surfaces primary risk drivers (contract, tenure, internet service, billing).
- **Side-by-Side Model Comparison:** Full comparison table across Accuracy, Precision, Recall, F1, and ROC-AUC.

---

## 🤖 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Optimal Threshold |
|---|---|---|---|---|---|---|
| 🌲 **Random Forest** | **76.6%** | **54.1%** | **78.1%** | **0.639** | **0.844** | `0.53` |
| 🌿 **Decision Tree** | 74.6% | 51.4% | 78.3% | **0.621** | 0.816 | `0.45` |
| 📈 **Logistic Regression** | 73.9% | 50.5% | **80.0%** | 0.619 | 0.838 | `0.48` |
| 🔍 **KNN** | **78.1%** | **58.3%** | 62.3% | 0.602 | 0.827 | `0.46` |

---

## 📁 Repository Structure

```
├── app.py                                # Main Streamlit Web Application
├── churn_prediction.ipynb                # Interactive Jupyter Notebook with all models & plots
├── train_model.py                        # Training & hyperparameter tuning script
├── requirements.txt                      # Deployment dependencies
├── .streamlit/
│   └── config.toml                       # Streamlit UI light theme configuration
├── model/                                # Saved ML Pipelines & Metadata
│   ├── lr_model.pkl
│   ├── dt_model.pkl
│   ├── rf_model.pkl
│   ├── knn_model.pkl
│   └── model_meta.json
├── figures/                              # Generated high-resolution plots & tree diagrams
└── Customer_Churn_Prediction_Report.docx # Complete mini project report
```

---

## 🛠️ Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Cloud (Free Hosting)

1. Push this repository to **GitHub**.
2. Sign in to **[share.streamlit.io](https://share.streamlit.io)** using GitHub.
3. Click **"New app"** and select:
   - **Repository:** `your-username/your-repo-name`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**
