from datetime import datetime
from typing import Any, Dict

import requests
from fastapi import HTTPException


def fetch_nasa_data(latitude: float, longitude: float) -> Dict[str, Any]:
    end_year = datetime.now().year - 1

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS2M",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": "20000101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch NASA POWER data: {str(exc)}",
        ) from exc

    if "properties" not in data or "parameter" not in data["properties"]:
        raise HTTPException(
            status_code=500,
            detail="Unexpected NASA POWER response format.",
        )

    return data