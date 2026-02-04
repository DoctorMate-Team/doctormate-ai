"""
Main application file with API routes.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from dotenv import load_dotenv
from typing import Optional

from app.models import (
    ApiResponse,
    SymptomsRequest,
    SymptomsResponseData,
    SkinLesionResponseData
)
from app.services import get_skin_service, get_symptoms_service
from app.utils import preprocess_image

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="DoctorMate AI",
    description="AI-powered medical assistant for skin lesion and symptom analysis",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "DoctorMate AI",
        "endpoints": ["/ai/skin/check", "/ai/symptoms/check"]
    }


@app.post("/ai/skin/check", response_model=ApiResponse[SkinLesionResponseData])
async def skin_check(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Analyze uploaded skin lesion image.
    
    Args:
        file: Uploaded image file
        authorization: Optional Bearer token for DoctorMate API (to fetch specialty/doctors)
        
    Returns:
        ApiResponse with skin lesion analysis results
    """
    try:
        # Preprocess image
        image_array = await preprocess_image(file)
        
        # Extract token from Authorization header
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
        
        # Get prediction from service (now async)
        skin_service = get_skin_service()
        prediction = await skin_service.predict(image_array, token)
        
        return ApiResponse(
            code=200,
            message="Skin lesion analysis completed successfully",
            data=prediction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/symptoms/check", response_model=ApiResponse[SymptomsResponseData])
async def symptoms_check(
    request: SymptomsRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Analyze user symptoms using AI.
    
    Args:
        request: SymptomsRequest with symptom description
        authorization: Optional Bearer token for DoctorMate API (to fetch specialty/doctors)
        
    Returns:
        ApiResponse with symptom analysis results
    """
    try:
        # Extract token from Authorization header
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "")
        
        # Get analysis from service
        symptoms_service = get_symptoms_service()
        analysis = await symptoms_service.analyze_symptoms(request.symptoms, token)
        
        return ApiResponse(
            code=200,
            message="Symptom analysis completed successfully",
            data=analysis
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


