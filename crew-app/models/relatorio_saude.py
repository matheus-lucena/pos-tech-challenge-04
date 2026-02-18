from typing import List
from pydantic import BaseModel, Field


class HealthReport(BaseModel):
    biometric_analysis: str = Field(..., description="SageMaker model result")
    emotional_analysis: str = Field(..., description="Audio/transcription analysis")
    maternal_analysis: str = Field(default="Not provided", description="Maternal signal analysis (MHR, variability, classification)")
    final_risk: str = Field(..., description="Final risk classification")
    recommendations: List[str] = Field(..., description="List of suggested actions")

