"""
Service layer for symptom analysis using Gemini API.
"""

import os
import json
from google import genai
from typing import Dict
from app.services.doctormate_api_service import (
    get_doctormate_api_service,
    SpecialtyMapper
)


class SymptomsAnalysisService:
    """Service for analyzing symptoms using Gemini API."""

    SYSTEM_PROMPT = """
You are a medical symptom analysis assistant.

Your task is to analyze the user's symptoms and return a structured medical assessment.

STRICT OUTPUT RULES:
- You MUST return ONLY valid JSON.
- Do NOT include explanations, markdown, or text outside the JSON.
- The JSON must strictly follow the schema below.
- If information is uncertain, make a reasonable assumption based on common medical knowledge.
- Confidence must be an integer between 0 and 100.

JSON SCHEMA:

{
  "possible_diagnosis": "string",
  "confidence": integer (0-100),
  "severity": "Mild" | "Moderate" | "Severe",
  "description": "string explaining the reasoning",
  "recommendations": ["string", "string"],
  "emergency_care": "string explaining when to seek urgent care",
  "disclaimer": "This is an AI-generated assessment and not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
}

RULES:
- possible_diagnosis: the most likely condition based on symptoms.
- confidence: likelihood estimate from 0–100.
- severity:
    Mild → minor symptoms
    Moderate → medical consultation recommended
    Severe → urgent medical care may be required
- recommendations: actionable suggestions
- emergency_care: describe red-flag symptoms requiring urgent care
- disclaimer MUST remain EXACTLY as written

Return JSON only.
"""

    def __init__(self):
        """Initialize Gemini client."""
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=self.api_key)

    async def analyze_symptoms(self, symptoms: str, token: str = None) -> Dict:
        """
        Analyze symptoms using Gemini API.
        """

        try:

            prompt = f"{self.SYSTEM_PROMPT}\n\nUser symptoms: {symptoms}"

            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2
                }
            )

            # -------- SAFE PARSING --------
            try:

                text = getattr(response, "text", None)

                if not text:
                    text = response.candidates[0].content.parts[0].text

                text = text.strip()

                # remove markdown json blocks
                if text.startswith("```"):
                    text = text.split("```")[1]

                result = json.loads(text)

            except Exception:
                print("JSON Parse Error:", response)

                result = {
                    "possible_diagnosis": "Unknown",
                    "confidence": 0,
                    "severity": "Moderate",
                    "description": "AI response parsing failed",
                    "recommendations": [
                        "Consult a healthcare provider"
                    ],
                    "emergency_care": "Seek urgent care if symptoms worsen.",
                    "disclaimer": "This is an AI-generated assessment and not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
                }

            # -------- FETCH SPECIALTY & DOCTORS --------

            specialty = {}
            recommended_doctors = []

            if token:

                diagnosis = result.get("possible_diagnosis", "")

                specialty_id = SpecialtyMapper.get_specialty_for_symptoms(
                    symptoms,
                    diagnosis
                )

                if specialty_id:

                    try:

                        api_service = get_doctormate_api_service(token)

                        specialty = await api_service.get_specialty(specialty_id) or {}

                        recommended_doctors = await api_service.get_recommended_doctors(
                            specialty_id,
                            limit=3
                        )

                    except Exception as e:
                        print("DoctorMate API error:", str(e))

            # attach results
            result["specialty"] = specialty
            result["recommended_doctors"] = recommended_doctors

            return result

        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


# -------- SINGLETON --------

_symptoms_service = None


def get_symptoms_service() -> SymptomsAnalysisService:
    """Return singleton instance."""

    global _symptoms_service

    if _symptoms_service is None:
        _symptoms_service = SymptomsAnalysisService()

    return _symptoms_service
