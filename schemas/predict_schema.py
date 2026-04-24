from datetime import date
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    prediction_type: Literal["monthly", "yearly"]
    start: date
    end: date

    @model_validator(mode="after")
    def validate_dates(self):
        if self.end < self.start:
            raise ValueError("end date must be after or equal start date")
        return self


class PredictResponse(BaseModel):
    status: str
    prediction_type: str
    start: date
    end: date
    prediction: Dict[str, Any]
    message: Optional[str] = None