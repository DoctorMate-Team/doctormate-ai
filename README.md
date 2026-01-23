# DoctorMate AI

AI-powered medical assistant for skin lesion analysis and symptom checking.

## Features

- 🔬 **Skin Lesion Analysis**: Upload images of skin lesions for AI-powered classification
- 🩺 **Symptom Checker**: Analyze symptoms using GPT-4o-mini for preliminary medical assessment
- 📊 **Standardized API Responses**: Consistent response format across all endpoints
- 🏗️ **Clean Architecture**: Service layer pattern with separation of concerns

## Tech Stack

- **FastAPI**: Modern web framework for building APIs
- **ONNX Runtime**: High-performance inference for skin lesion classification
- **OpenAI GPT-4o-mini**: Advanced symptom analysis
- **Pydantic**: Data validation using Python type annotations
- **Python 3.12**: Latest Python features

## Project Structure

```
doctor_mate_ai/
├── app/
│   ├── main.py              # API routes and endpoints
│   ├── models.py            # Pydantic models for request/response
│   ├── utils.py             # Utility functions (image preprocessing)
│   └── services/
│       ├── skin_rules.json      # Skin Rules - Knowledge Based
│       ├── skin_service.py      # Skin lesion analysis service
│       └── symptoms_service.py  # Symptom analysis service
├── models/
│   └── MobileNetV2_best.onnx    # ONNX model for skin lesion classification
├── .env                     # Environment variables (not in git)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.12+
- pip
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd doctor_mate_ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   
   Windows (PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   Linux/Mac:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

Start the development server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Unified Response Structure

All endpoints return responses in a consistent format:

```json
{
  "code": 200,
  "message": "Success message",
  "data": {
    "possible_diagnosis": "Diagnosis name",
    "confidence": 85,  // Integer percentage (0-100)
    "severity": "Mild" | "Moderate" | "Severe",
    "description": "Detailed medical description",
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "emergency_care": "When to seek immediate medical attention",
    "additional_info": {},  // Optional: extra fields (skin lesions only)
    "disclaimer": "Medical disclaimer message"
  }
}
```

## API Endpoints

### 1. Health Check

**GET** `/`

Returns service status and available endpoints.

**Response:**
```json
{
  "status": "healthy",
  "service": "DoctorMate AI",
  "endpoints": ["/ai/skin/check", "/ai/symptoms/check"]
}
```

### 2. Skin Lesion Analysis

**POST** `/ai/skin/check`

Analyze uploaded skin lesion image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "code": 200,
  "message": "Skin lesion analysis completed successfully",
  "data": {
    "possible_diagnosis": "Melanocytic Nevi (Moles)",
    "confidence": 95,
    "severity": "Mild",
    "description": "Melanocytic nevi, commonly known as moles, are benign growths of melanocytes...",
    "recommendations": [
      "Routine monitoring for changes (ABCDE rule)",
      "Annual skin examination by dermatologist",
      "Monthly self-skin examinations",
      "Sun protection to prevent new moles"
    ],
    "emergency_care": "Seek evaluation if a mole shows signs of melanoma...",
    "additional_info": {
      "diagnosis_code": "nv",
      "risk_factors": ["Genetics (hereditary)", "Sun exposure", "Fair skin"],
      "symptoms": ["Round or oval shape", "Even color", "Clear borders"],
      "prognosis": "Excellent. Most moles remain benign throughout life.",
      "treatment_options": ["Observation (most common)", "Surgical excision"],
      "all_probabilities": {
        "akiec": 0.1,
        "bcc": 0.2,
        "bkl": 1.0,
        "df": 0.5,
        "mel": 0.2,
        "nv": 95.0,
        "vasc": 3.0
      }
    },
    "disclaimer": "This is an AI-generated assessment and not a substitute for professional medical advice..."
  }
}
```

**Skin Lesion Classes:**
- `akiec`: Actinic keratoses and intraepithelial carcinoma (Moderate severity)
- `bcc`: Basal cell carcinoma (Moderate severity)
- `bkl`: Benign keratosis-like lesions (Mild severity)
- `df`: Dermatofibroma (Mild severity)
- `mel`: Melanoma (Severe severity - requires urgent attention)
- `nv`: Melanocytic nevi (Moles) (Mild severity)
- `vasc`: Vascular lesions (Mild severity)

### Knowledge Base

The skin lesion analysis uses a comprehensive JSON knowledge base ([skin_rules.json](app/services/skin_rules.json)) containing:
- Detailed medical descriptions
- Risk factors
- Symptoms and signs
- Treatment recommendations
- Emergency care guidelines
- Prognosis information
- Treatment options

This structured approach ensures consistent, medically-informed responses similar to the symptom checker.

### 3. Symptom Analysis

**POST** `/ai/symptoms/check`

Analyze user symptoms using AI.

**Request:**
```json
{
  "symptoms": "Sore throat, fever, headache, runny nose for 3 days"
}
```

**Response:**
```json
{
  "code": 200,
  "message": "Symptom analysis completed successfully",
  "data": {
    "possible_diagnosis": "Common Cold (Upper Respiratory Infection)",
    "confidence": 85,
    "severity": "Mild",
    "description": "Based on your symptoms of sore throat, fever, headache, and runny nose lasting 3 days, this appears to be a common cold...",
    "recommendations": [
      "Rest and stay hydrated with plenty of fluids",
      "Use over-the-counter pain relievers for fever and headache",
      "Gargle with warm salt water for throat relief",
      "Use a humidifier to ease congestion"
    ],
    "emergency_care": "Seek immediate care if fever exceeds 103°F, symptoms worsen after 7 days, or difficulty breathing occurs.",
    "disclaimer": "This is an AI-generated assessment and not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."
  }
}
```

## Testing

Use the interactive documentation at http://localhost:8000/docs to test endpoints with a user-friendly interface.

## Medical Disclaimer

⚠️ **IMPORTANT**: This application is for educational and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Deployment

### Deploy to Railway

1. **Fork/Clone the repository**

2. **Connect to Railway**
   - Go to [Railway](https://railway.app/)
   - Create a new project
   - Connect your GitHub repository

3. **Configure Environment Variables**
   
   Add the following environment variable in Railway dashboard:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Deployment Settings** (Automatically configured via `railway.json`):
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Healthcheck Path**: `/`
   - **Restart Policy**: On Failure
   - **Max Retries**: 10

5. **Deploy**
   - Push to `main` branch to trigger automatic deployment
   - Railway will automatically detect the Python app and install dependencies
   - Access your deployed API at the Railway-provided URL

### Environment Variables Required

```
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: Never commit your `.env` file to version control. The `.gitignore` file is configured to exclude it.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Authors

[Add author information here]
