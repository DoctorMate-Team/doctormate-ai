"""
Service for interacting with DoctorMate external API.
"""
import os
import httpx
from typing import List, Dict, Optional
from enum import Enum


class SpecialtyMapper:
    """Maps medical conditions to specialty IDs."""
    
    # Skin lesion to specialty mapping
    SKIN_LESION_MAP = {
        "mel": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology - Melanoma
        "bcc": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology - Basal cell carcinoma
        "akiec": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology - Actinic keratoses
        "bkl": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology - Benign keratosis
        "df": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology - Dermatofibroma
        "nv": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology - Melanocytic nevi
        "vasc": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology - Vascular lesions
    }
    
    # Symptom keywords to specialty mapping
    SYMPTOM_KEYWORDS_MAP = {
        "heart|chest pain|palpitation|cardiac": "a1111111-a1a1-a1a1-a1a1-a1a1a1a1a1a1",  # Cardiology
        "brain|headache|seizure|neurological|nerve": "5b05c49a-288f-48f3-b684-6d505c58d276",  # Neurology
        "skin|rash|acne|eczema|dermatological": "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2",  # Dermatology
        "child|infant|pediatric|baby": "bb79c512-c722-4e5a-a1fc-c9699359b636",  # Pediatrics
    }
    
    @staticmethod
    def get_specialty_for_skin_lesion(diagnosis_code: str) -> str:
        """Get specialty ID for skin lesion diagnosis."""
        return SpecialtyMapper.SKIN_LESION_MAP.get(
            diagnosis_code,
            "b2222222-b2b2-b2b2-b2b2-b2b2b2b2b2b2"  # Default to Dermatology
        )
    
    @staticmethod
    def get_specialty_for_symptoms(symptoms: str, diagnosis: str) -> str:
        """Get specialty ID based on symptoms and diagnosis."""
        symptoms_lower = symptoms.lower()
        diagnosis_lower = diagnosis.lower()
        combined = f"{symptoms_lower} {diagnosis_lower}"
        
        for keywords, specialty_id in SpecialtyMapper.SYMPTOM_KEYWORDS_MAP.items():
            for keyword in keywords.split("|"):
                if keyword in combined:
                    return specialty_id
        
        # Default to general specialty if no match
        return "fa12fdd9-0a6a-4330-814c-17fd31fbd637"


class DoctorMateAPIService:
    """Service for fetching specialties and doctors from DoctorMate API."""
    
    BASE_URL = "https://doctormate.runasp.net/api"
    
    def __init__(self, token: str):
        """Initialize the service with API token.
        
        Args:
            token: Bearer token for DoctorMate API authentication
        """
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def get_specialty(self, specialty_id: str) -> Optional[Dict]:
        """
        Get specialty details by ID.
        
        Args:
            specialty_id: Specialty UUID
            
        Returns:
            Specialty information or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/Specialties",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    specialties = data.get("data", [])
                    
                    # Find the specialty by ID
                    for specialty in specialties:
                        if specialty.get("id") == specialty_id:
                            return {
                                "id": specialty.get("id"),
                                "name": specialty.get("name"),
                                "description": specialty.get("description"),
                                "imageUrl": specialty.get("imageUrl")
                            }
                return None
                
        except Exception as e:
            print(f"Error fetching specialty: {str(e)}")
            return None
    
    async def get_recommended_doctors(
        self, 
        specialty_id: str, 
        limit: int = 3
    ) -> List[Dict]:
        """
        Get recommended doctors for a specialty.
        
        Args:
            specialty_id: Specialty UUID
            limit: Maximum number of doctors to return
            
        Returns:
            List of recommended doctors
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.BASE_URL}/Specialties/{specialty_id}/doctors",
                    headers=self.headers,
                    params={"page": 1, "limit": limit}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    doctors = data.get("data", {}).get("doctors", [])
                    
                    # Format doctor information
                    return [
                        {
                            "id": doctor.get("id"),
                            "fullName": doctor.get("fullName"),
                            "imageUrl": doctor.get("imageUrl"),
                            "consultationFee": doctor.get("consultationFee"),
                            "address": doctor.get("address"),
                            "workingTime": doctor.get("workingTime"),
                            "qualifications": doctor.get("qualifications"),
                        }
                        for doctor in doctors[:limit]
                    ]
                    
                return []
                
        except Exception as e:
            print(f"Error fetching doctors: {str(e)}")
            return []


def get_doctormate_api_service(token: str) -> DoctorMateAPIService:
    """Get instance of DoctorMateAPIService with provided token.
    
    Args:
        token: Bearer token for DoctorMate API authentication
        
    Returns:
        DoctorMateAPIService instance
    """
    return DoctorMateAPIService(token)
