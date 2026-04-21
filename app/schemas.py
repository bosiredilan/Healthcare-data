from pydantic import BaseModel, Field, field_validator

VALID_GENDERS = ["Male", "Female"]
VALID_BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
VALID_MEDICAL_CONDITIONS = [
    "Diabetes",
    "Hypertension",
    "Pneumonia",
    "Malaria",
    "Typhoid",
    "HIV/AIDS",
]
VALID_INSURANCE_PROVIDERS = [
    "NHIF",
    "Jubilee Insurance",
    "AAR Insurance",
    "CIC Insurance",
    "Britam",
]
VALID_ADMISSION_TYPES = ["Elective", "Emergency", "Urgent"]
VALID_MEDICATIONS = [
    "Metformin",
    "Amoxicillin",
    "Coartem",
    "Tenofovir",
    "Sulphadoxine-Pyrimethamine",
]


class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, examples=[34])
    gender: str = Field(..., examples=["Female"])
    blood_type: str = Field(..., examples=["B+"])
    medical_condition: str = Field(..., examples=["Malaria"])
    insurance_provider: str = Field(..., examples=["NHIF"])
    billing_amount: float = Field(..., gt=0, examples=[45500.00])
    admission_type: str = Field(..., examples=["Emergency"])
    medication: str = Field(..., examples=["Coartem"])

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in VALID_GENDERS:
            raise ValueError(f"gender must be one of {VALID_GENDERS}")
        return v

    @field_validator("blood_type")
    @classmethod
    def validate_blood_type(cls, v: str) -> str:
        if v not in VALID_BLOOD_TYPES:
            raise ValueError(f"blood_type must be one of {VALID_BLOOD_TYPES}")
        return v

    @field_validator("medical_condition")
    @classmethod
    def validate_medical_condition(cls, v: str) -> str:
        if v not in VALID_MEDICAL_CONDITIONS:
            raise ValueError(f"medical_condition must be one of {VALID_MEDICAL_CONDITIONS}")
        return v

    @field_validator("insurance_provider")
    @classmethod
    def validate_insurance_provider(cls, v: str) -> str:
        if v not in VALID_INSURANCE_PROVIDERS:
            raise ValueError(f"insurance_provider must be one of {VALID_INSURANCE_PROVIDERS}")
        return v

    @field_validator("admission_type")
    @classmethod
    def validate_admission_type(cls, v: str) -> str:
        if v not in VALID_ADMISSION_TYPES:
            raise ValueError(f"admission_type must be one of {VALID_ADMISSION_TYPES}")
        return v

    @field_validator("medication")
    @classmethod
    def validate_medication(cls, v: str) -> str:
        if v not in VALID_MEDICATIONS:
            raise ValueError(f"medication must be one of {VALID_MEDICATIONS}")
        return v


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model_version: str
    context: str = "KE"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset_version: str
    version: str


class RetrainResponse(BaseModel):
    status: str
    algorithm: str
    macro_f1: float
    message: str