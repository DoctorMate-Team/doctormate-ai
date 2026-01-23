"""
Utility functions for image preprocessing.
"""
import numpy as np
from PIL import Image
from fastapi import UploadFile
import io


async def preprocess_image(file: UploadFile) -> np.ndarray:
    """
    Preprocess uploaded image for model inference.
    
    Args:
        file: Uploaded image file
        
    Returns:
        np.ndarray: Preprocessed image array in NHWC format (batch, height, width, channels)
    """
    # Read file contents
    contents = await file.read()
    
    # Open and convert to RGB
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    
    # Add batch dimension (model expects NHWC format: batch, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

