from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from app.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    blood_type = Column(String(5), nullable=False)
    medical_condition = Column(String(60), nullable=False)
    insurance_provider = Column(String(60), nullable=False)
    billing_amount = Column(Float, nullable=False)
    admission_type = Column(String(20), nullable=False)
    medication = Column(String(60), nullable=False)
    test_results = Column(String(20), nullable=True)
    county = Column(String(40), nullable=True)
    dataset_version = Column(String(5), default="KE")
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, nullable=True)
    predicted_result = Column(String(20), nullable=False)
    probability_normal = Column(Float, nullable=False)
    probability_abnormal = Column(Float, nullable=False)
    probability_inconclusive = Column(Float, nullable=False)
    model_version = Column(String(40), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version_tag = Column(String(40), nullable=False, unique=True)
    algorithm = Column(String(30), nullable=False)
    macro_f1 = Column(Float, nullable=False)
    dataset_version = Column(String(5), default="KE")
    is_active = Column(Integer, default=1)
    trained_at = Column(DateTime, default=datetime.utcnow)