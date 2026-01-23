"""
Service layer for symptom analysis using OpenAI.
"""
import os
import json
from openai import AsyncOpenAI
from typing import Dict


class SymptomsAnalysisService:
    """Service for analyzing symptoms using OpenAI API."""
    
    # System prompt for medical assessment
    SYSTEM_PROMPT = """You are a medical symptom checker assistant. 
Analyze the user's symptoms and provide a structured medical assessment.

IMPORTANT: You must respond ONLY with valid JSON in this exact format:
{
  "possible_diagnosis": "string",
  "confidence": number (0-100),
  "severity": "Mild" or "Moderate" or "Severe",
  "description": "string",
  "recommendations": ["string", "string", ...],
  "emergency_care": "string",
  "disclaimer": "This is an AI-generated assessment and not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
}

Base your analysis on common medical knowledge. Be cautious and recommend professional consultation when appropriate."""
    
    def __init__(self):
        """Initialize the service."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    async def analyze_symptoms(self, symptoms: str) -> Dict:
        """
        Analyze symptoms using OpenAI API.
        
        Args:
            symptoms: User's symptom description
            
        Returns:
            dict: Structured medical assessment
            
        Raises:
            Exception: If OpenAI API call fails
        """
        client = AsyncOpenAI(api_key=self.api_key)
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"User symptoms: {symptoms}"}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


# Singleton instance
_symptoms_service = None

def get_symptoms_service() -> SymptomsAnalysisService:
    """Get singleton instance of SymptomsAnalysisService."""
    global _symptoms_service
    if _symptoms_service is None:
        _symptoms_service = SymptomsAnalysisService()
    return _symptoms_service
