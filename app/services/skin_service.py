"""
Service layer for skin lesion analysis.
"""
import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
from typing import Dict
from app.services.doctormate_api_service import (
    get_doctormate_api_service,
    SpecialtyMapper
)


class SkinAnalysisService:
    """Service for analyzing skin lesions using ONNX model."""
    
    # Class constants
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    
    def __init__(self, model_path: str = "models/MobileNetV2_best.onnx"):
        """Initialize the service with ONNX model and knowledge base."""
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)
        
        # Load knowledge base
        knowledge_base_path = Path(__file__).parent / "skin_rules.json"
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
    async def predict(self, image_array: np.ndarray, token: str = None) -> dict:
        """
        Predict skin lesion type from preprocessed image array.
        
        Args:
            image_array: Preprocessed image array in NHWC format
            token: Optional DoctorMate API bearer token for fetching specialty/doctors
            
        Returns:
            dict: Prediction results with diagnosis, confidence, severity, etc.
        """
        # Run inference
        inputs = {self.session.get_inputs()[0].name: image_array}
        outputs = self.session.run(None, inputs)
        predictions = outputs[0]
        
        # Format response
        return await self._format_prediction(predictions, token)
    
    async def _format_prediction(self, predictions: np.ndarray, token: str = None) -> dict:
        """Format raw model predictions into structured response with knowledge base."""
        probs = predictions[0]
        idx = int(np.argmax(probs))
        
        diagnosis = self.CLASS_NAMES[idx]
        confidence = float(probs[idx])
        
        # Get detailed information from knowledge base
        lesion_info = self.knowledge_base.get(diagnosis, {})
        
        # Get specialty and doctors from DoctorMate API if token provided
        specialty_id = SpecialtyMapper.get_specialty_for_skin_lesion(diagnosis)
        specialty = {}
        recommended_doctors = []
        
        if token:
            try:
                api_service = get_doctormate_api_service(token)
                specialty = await api_service.get_specialty(specialty_id) or {}
                recommended_doctors = await api_service.get_recommended_doctors(specialty_id, limit=3)
            except Exception as e:
                print(f"Error fetching specialty/doctors: {str(e)}")
        
        return {
            "possible_diagnosis": lesion_info.get("name", "Unknown"),
            "confidence": int(round(confidence * 100)),  # Convert to percentage
            "severity": lesion_info.get("severity", "Mild"),
            "description": lesion_info.get("description", "No description available."),
            "recommendations": lesion_info.get("recommendations", []),
            "emergency_care": lesion_info.get("emergency_care", "Consult a healthcare provider if concerned."),
            "specialty": specialty,
            "recommended_doctors": recommended_doctors,
            "additional_info": {
                "diagnosis_code": diagnosis,
                "risk_factors": lesion_info.get("risk_factors", []),
                "symptoms": lesion_info.get("symptoms", []),
                "prognosis": lesion_info.get("prognosis", "Please consult a dermatologist for proper assessment."),
                "treatment_options": lesion_info.get("treatment_options", []),
                "all_probabilities": {
                    self.CLASS_NAMES[i]: round(float(probs[i]) * 100, 1)
                    for i in range(len(self.CLASS_NAMES))
                }
            },
            "disclaimer": "This is an AI-generated assessment and not a substitute for professional medical advice. Please consult a dermatologist for proper diagnosis and treatment."
        }


# Singleton instance
_skin_service = None

def get_skin_service() -> SkinAnalysisService:
    """Get singleton instance of SkinAnalysisService."""
    global _skin_service
    if _skin_service is None:
        _skin_service = SkinAnalysisService()
    return _skin_service
