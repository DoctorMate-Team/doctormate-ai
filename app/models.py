"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Generic, TypeVar
from enum import Enum

T = TypeVar('T')


class SeverityLevel(str, Enum):
    mild = "Mild"
    moderate = "Moderate"
    severe = "Severe"


class ApiResponse(BaseModel, Generic[T]):
    """Standardized API response format."""
    code: int
    message: str
    data: T


class SymptomsRequest(BaseModel):
    symptoms: str = Field(..., min_length=1)


class SymptomsResponseData(BaseModel):
    """Response data model for symptom analysis."""
    possible_diagnosis: str
    confidence: int = Field(..., ge=0, le=100)
    severity: SeverityLevel
    description: str
    recommendations: List[str]
    emergency_care: str
    disclaimer: str


class SkinLesionResponseData(BaseModel):
    """Response data model for skin lesion analysis."""
    possible_diagnosis: str = Field(..., description="Human-readable diagnosis name")
    confidence: int = Field(..., ge=0, le=100, description="Confidence percentage")
    severity: SeverityLevel
    description: str
    recommendations: List[str]
    emergency_care: str
    additional_info: Dict = Field(
        default_factory=dict,
        description="Additional medical information (diagnosis_code, risk_factors, symptoms, prognosis, treatment_options, probabilities)"
    )
    disclaimer: str