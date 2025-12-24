import uuid
import json
from flask import Blueprint, request, jsonify
from app.validator import require_token
from app.api.id_card_scan.algorithm import process_ocr_images, process_extract_images

api_v1_bp = Blueprint('api_v1', __name__)


@api_v1_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify(
        success=True,
        message='OCR API is running',
        status='healthy'
    ), 200


@api_v1_bp.route('/ocr', methods=['POST'])
@require_token
def ocr_endpoint():
    """
    OCR endpoint for ID card scanning.
    
    Expected request:
    - multipart/form-data
    - image_file: image file(s) to process
    - hard_token: authentication token (if required)
    
    Response:
    - JSON with OCR results
    """
    trace_id = str(uuid.uuid4())
    
    # Get uploaded files
    files = request.files.getlist('image_file')
    if not files or len(files) == 0:
        return jsonify(
            success=False,
            message='No image_file provided in request',
            error_code='MISSING_FILE',
            trace_id=trace_id
        ), 400
    
    # Get display_text_box option from form data (default: true)
    display_text_box = request.form.get('display_text_box', 'true').lower() == 'true'
    
    # Process images
    results, errors = process_ocr_images(files, display_text_box=display_text_box)
    
    # Prepare response
    response_data = {
        'results': results,
        'processed_count': len(results),
        'error_count': len(errors)
    }
    
    if errors:
        response_data['errors'] = errors
    
    return jsonify(
        success=len(errors) == 0,
        message='OCR processing completed' if len(results) > 0 else 'No files processed',
        trace_id=trace_id,
        data=response_data
    ), 200 if len(results) > 0 else 400


@api_v1_bp.route('/extract', methods=['POST'])
@require_token
def extract_endpoint():
    """
    Extract structured information from ID card images using OCR + LLM.
    
    Expected request:
    - multipart/form-data
    - image_file: image file(s) to process
    - config: JSON config with fields to extract
    - hard_token: authentication token (if required)
    
    Config format:
    {
        "info": {
            "field_key1": "Description in Vietnamese",
            "field_key2": "Another description"
        },
        "parallel": true/false,  # Use parallel extraction (optional)
        "batch_size": 3,         # Parallel batch size (optional)
        "llm_timeout": 30        # LLM query timeout in seconds (optional)
    }
    
    Response:
    - JSON with extracted fields for each image
    """
    trace_id = str(uuid.uuid4())
    
    # Get uploaded files
    files = request.files.getlist('image_file')
    if not files or len(files) == 0:
        return jsonify(
            success=False,
            message='No image_file provided in request',
            error_code='MISSING_FILE',
            trace_id=trace_id
        ), 400
    
    # Get config
    config_str = request.form.get('config')
    if not config_str:
        return jsonify(
            success=False,
            message='No config provided in request',
            error_code='MISSING_CONFIG',
            trace_id=trace_id
        ), 400
    
    try:
        config = json.loads(config_str)
    except json.JSONDecodeError as e:
        return jsonify(
            success=False,
            message=f'Invalid config JSON: {str(e)}',
            error_code='INVALID_CONFIG',
            trace_id=trace_id
        ), 400
    
    # Validate config
    info_dict = config.get('info', {})
    if not info_dict or not isinstance(info_dict, dict):
        return jsonify(
            success=False,
            message='Config must include "info" dict with field descriptions',
            error_code='INVALID_CONFIG',
            trace_id=trace_id
        ), 400
    
    # Extract options
    use_parallel = config.get('parallel', True)
    batch_size = config.get('batch_size', 3)
    llm_timeout = config.get('llm_timeout', 30)
    display_text_box = config.get('display_text_box', False)
    
    # Process images through OCR + LLM pipeline
    results, errors = process_extract_images(
        files,
        info_dict,
        use_parallel=use_parallel,
        batch_size=batch_size,
        llm_timeout=llm_timeout,
        display_text_box=display_text_box
    )
    
    # Prepare response
    response_data = {
        'results': results,
        'processed_count': len(results),
        'error_count': len(errors)
    }
    
    if errors:
        response_data['errors'] = errors
    
    return jsonify(
        success=len(errors) == 0,
        message='Information extraction completed' if len(results) > 0 else 'No files processed',
        trace_id=trace_id,
        data=response_data
    ), 200 if len(results) > 0 else 400


@api_v1_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify(
        success=False,
        message='Endpoint not found',
        error_code='NOT_FOUND'
    ), 404


@api_v1_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify(
        success=False,
        message='Internal server error',
        error_code='INTERNAL_ERROR'
    ), 500
