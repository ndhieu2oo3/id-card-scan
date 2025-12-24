import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_file) -> np.ndarray:
    """
    Load and preprocess image from file object.
    
    Args:
        image_file: File object from Flask request
        
    Returns:
        np.ndarray: Image in BGR format (OpenCV compatible)
        
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        # Load image from file object
        img = Image.open(image_file.stream if hasattr(image_file, 'stream') else image_file)
        img = img.convert('RGB')
        
        # Convert to numpy array
        arr = np.asarray(img, dtype='uint8')
        
        # Convert RGB -> BGR (OpenCV format)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        return arr
    except Exception as e:
        raise ValueError(f'Failed to preprocess image: {str(e)}')
