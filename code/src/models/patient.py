"""Patient data models aligned with FHIR R4 standard."""

from __future__ import annotations
from datetime import date, datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class Symptom(BaseModel):
    name: str
    duration_days: Optional[int] = None
    severity: Severity = Severity.MODERATE
    description: Optional[str] = None


class Allergy(BaseModel):
    substance: str
    reaction: Optional[str] = None
    severity: Severity = Severity.MODERATE


class Medication(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[date] = None


class VitalSigns(BaseModel):
    temperature: Optional[float] = Field(None, description="Body temperature in Celsius")
    heart_rate: Optional[int] = Field(None, description="Beats per minute")
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None


class LabResult(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    is_abnormal: bool = False


class PatientInfo(BaseModel):
    """Core patient information for intake processing."""
    patient_id: Optional[str] = None
    name: str
    age: int
    gender: Gender
    chief_complaint: str
    symptoms: list[Symptom] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    current_medications: list[Medication] = Field(default_factory=list)
    vital_signs: Optional[VitalSigns] = None
    lab_results: list[LabResult] = Field(default_factory=list)
    raw_input: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
