import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # Device configuration
    DEVICE_GPU: int = int(os.environ.get('DEVICE_GPU', '-1'))
    
    # Authentication
    HARD_TOKEN: str = os.environ.get('HARD_TOKEN', '')
    
    # Model paths
    TEXT_REC_MODEL_PATH: str = os.environ.get(
        'TEXT_REC_MODEL_PATH',
        './app/vision/weights/rec_svtr_32x480_v0.onnx'
    )
    TEXT_REC_DICT_PATH: str = os.environ.get(
        'TEXT_REC_DICT_PATH',
        './app/vision/weights/vn_dict.txt'
    )
    TEXT_DET_PDPARAM_MODEL_PATH: str = os.environ.get(
        'TEXT_DET_PDPARAM_MODEL_PATH',
        './app/vision/weights/en_PP-OCRv3_det_infer'
    )
    
    # OCR parameters
    TEXT_REC_TARGET_SIZE: str = os.environ.get('TEXT_REC_TARGET_SIZE', '3,32,480')
    TEXT_REC_CONF_THRESHOLD: float = float(
        os.environ.get('TEXT_REC_CONF_THRESHOLD', '0.7')
    )
    TEXT_DET_CONF_THRESHOLD: float = float(
        os.environ.get('TEXT_DET_CONF_THRESHOLD', '0.3')
    )
    
    # Server
    HOST: str = os.environ.get('HOST', '0.0.0.0')
    PORT: int = int(os.environ.get('PORT', '5000'))
    DEBUG: bool = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    @property
    def rec_target_size(self) -> list:
        """Parse target size from string."""
        return [int(x) for x in self.TEXT_REC_TARGET_SIZE.split(',')]


settings = Settings()
