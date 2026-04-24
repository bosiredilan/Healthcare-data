

---

# 🏥 AfyaPredict KE: Kenyan Clinical Test Intelligence Platform

**AfyaPredict KE** is a production-grade machine learning system designed to bridge the gap between raw patient admission records and automated clinical test result predictions within the Kenyan healthcare context. Built on a FastAPI backend, the platform leverages a dataset of 55,000+ records to predict test outcomes as **Normal**, **Abnormal**, or **Inconclusive**.

The system is fully automated: it retrains every Saturday via GitHub Actions to ensure the model adapts to new data trends without human intervention.

## 🎯 Project Goal
In many Kenyan clinical settings, manual processing of admission data is time-consuming. **AfyaPredict KE** automates this: demographic and clinical data (including NHIF/local insurance data and local disease prevalence) are submitted via API, and the model returns a prediction with probability scores in milliseconds. This ensures faster triaging and data-driven clinical support.

## 🧬 System Architecture

1.  **Data Ingestion**: `scripts/ingest.py` pulls the primary healthcare dataset and prepares it for the Kenyan context.
2.  **Data Cleaning**: `scripts/clean.py` standardizes categorical columns, handles date formatting, and removes duplicates to ensure high data quality.
3.  **Data Storage**: Cleaned records are persisted in a PostgreSQL database using SQLAlchemy for robust data management.
4.  **ML Training**: `ml/train.py` encodes 8 features, scales numeric values, and compares **XGBoost** vs. **Random Forest**. The best model (typically Random Forest for this noise level) is saved as `model.joblib`.
5.  **Model Serving**: A FastAPI wrapper exposes a `/predict` endpoint. The model is held in-memory for sub-millisecond inference.
6.  **Automated Retraining**: A GitHub Actions cron job (`0 12 * * 6`) retrains the model every Saturday at noon UTC, ensuring zero-drift.
7.  **Frontend**: A localized web interface (Swa-English) allows users to input data and see real-time visualizations of the prediction probability.


## 🛠️ Technical Stack
| Layer | Tool |
| :--- | :--- |
| **API / Backend** | FastAPI, Uvicorn, Pydantic |
| **ML Frameworks** | Scikit-learn, XGBoost |
| **Data Handling** | Pandas, NumPy, Joblib |
| **Database** | PostgreSQL, SQLAlchemy |
| **Automation** | GitHub Actions, Apache Airflow 3 |
| **Deployment** | Render.com |

## 📊 Performance & Results
* **Class Balance**: Balanced 3-way split (~18k records each for Normal, Abnormal, and Inconclusive).
* **Random Forest Accuracy**: **37.7%** (Exceeds the 33.3% random baseline for a 3-class problem).
* **Response Time**: < 50ms per prediction.
* **Optimization**: Artifacts are compressed to **5.1 MB** to fit within GitHub/Render deployment constraints while maintaining performance.

## 📸 Screenshots
### Prediction UI — Live Result
<img width="1038" height="845" alt="health predictor web" src="https://github.com/user-attachments/assets/5a906ba6-8974-4907-b744-92d25f18bd54" />

*Patient data submitted: Age 80 · Female · B+ · Emergency · Billing KES 45,500 · NHIF · Diabetes · SP — Predicted **Abnormal** with 49% confidence.*

## 🧬 Dataset Features (Localized for Kenya)
| Feature | Type | Values / Examples |
| :--- | :--- | :--- |
| **Age** | Numeric | 1 – 120 |
| **Gharama (Billing)** | Numeric | Continuous (**KES**) |
| **Gender** | Categorical | Male, Female |
| **Blood Type** | Categorical | A+, A−, B+, B−, AB+, AB−, O+, O− |
| **Medical Condition** | Categorical | **HIV/AIDS, Malaria, Diabetes, Pneumonia, Hypertension, Typhoid** |
| **Insurance Provider**| Categorical | **Britam, NHIF, Jubilee, AAR, CIC Insurance** |
| **Admission Type** | Categorical | Emergency, Elective, Urgent |
| **Medication** | Categorical | **Coartem, Amoxicillin, Metformin, Tenofovir, SP** |

## 🧠 Key Design Decisions
* **Random Forest over Logistic Regression**: Better at handling the non-linear, noisy relationships found in synthetic healthcare data and provides better probability calibration for the UI.
* **StandardScaler & LabelEncoding**: Encoders are saved as separate artifacts to ensure the transformation applied during inference exactly matches the training state.
* **Localized Context**: The model was adapted to use Kenyan currency (KES) and specific medical conditions/medications prevalent in the region to make the tool relevant for local practitioners.
* **Stateless Deployment**: Since Render's filesystem is ephemeral, GitHub Actions acts as the "long-term memory" by committing the retrained model back to the repository.

## 📂 Project Structure
```text
├── .github/workflows/retrain.yml  # Automated Saturday retraining
├── app/
│   ├── main.py                    # FastAPI Entry point
│   ├── routes.py                  # API Endpoints (/predict, /retrain)
│   └── model_loader.py            # Singleton for loading .joblib files
├── data/
│   └── cleaned_healthcare.csv      # Localized dataset
├── database/
│   ├── models.py                  # SQLAlchemy ORM
│   └── queries.sql                # Table Schemas
├── ml/
│   ├── train.py                   # Model selection & training logic
│   └── preprocess.py              # Scaling and Encoding
├── models/
│   ├── model.joblib               # Best classifier artifact
│   └── encoders.joblib            # Transformation artifacts
└── frontend/
    └── index.html                 # Swa-English Prediction UI
```

## ⚙️ Installation & Setup
1.  **Clone & Environment**:
    ```bash
    git clone https://github.com/bosiredilan/Healthcare-data.git
    cd Healthcare-data
    uv venv
    source .venv/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Run Locally**:
    ```bash
    # Initial setup
    python scripts/clean.py
    python ml/train.py
    
    # Start Server
    uvicorn app.main:app --reload
    ```
4.  **Access Docs**: Navigate to `http://localhost:8000/docs` to test the API via Swagger.

---
**Note:** *This dataset is based on synthetic data. The accuracy reflects the model's ability to find signal in a high-noise environment. Always consult a medical professional for actual clinical diagnosis.*
