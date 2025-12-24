#!/usr/bin/env python3
"""Main entry point for ID Card OCR API."""

import sys
import os
import logging

# Suppress PaddleOCR warnings
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.ERROR)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extentions import settings

if __name__ == '__main__':
    app = create_app()
    
    print(f'Starting ID Card OCR API')
    print(f'Host: {settings.HOST}')
    print(f'Port: {settings.PORT}')
    print(f'Debug: {settings.DEBUG}')
    print(f'Device GPU: {settings.DEVICE_GPU}')
    
    try:
        app.run(
            host=settings.HOST,
            port=settings.PORT,
            debug=settings.DEBUG,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print('\nShutting down...')
        sys.exit(0)
