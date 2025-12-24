"""
OCR post-processing and text structuring algorithms.
Complete processing pipeline for OCR and LLM-based extraction.
"""

import numpy as np
import json
import re
from typing import List, Dict, Tuple, Optional
from app.api.id_card_scan.utils import preprocess_image
from app.vision.ocr.ocr_service import get_ocr_system
from app.vision.ocr.llm_extractor import LLMClient


def sort_boxes_by_position(boxes: List[List], texts: List[str]) -> Tuple[List[List], List[str]]:
    """
    Sort OCR boxes and texts by reading order (top-to-bottom, left-to-right).
    
    Args:
        boxes: List of bounding boxes [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        texts: List of recognized texts
        
    Returns:
        Tuple of (sorted_boxes, sorted_texts)
    """
    if not boxes or not texts:
        return boxes, texts
    
    # Calculate center y coordinate for each box (for top-to-bottom sorting)
    box_centers = []
    for box in boxes:
        # Extract y coordinates from all corners
        y_coords = [point[1] for point in box]
        center_y = np.mean(y_coords)
        x_coords = [point[0] for point in box]
        center_x = np.mean(x_coords)
        box_centers.append((center_y, center_x))
    
    # Sort by y first (top-to-bottom), then by x (left-to-right)
    sorted_indices = sorted(range(len(box_centers)), key=lambda i: (box_centers[i][0], box_centers[i][1]))
    
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_texts = [texts[i] for i in sorted_indices]
    
    return sorted_boxes, sorted_texts


def group_text_by_lines(boxes: List[List], texts: List[str], 
                       y_threshold: float = 20) -> List[List[Dict]]:
    """
    Group OCR results into lines based on y-coordinate proximity.
    
    Args:
        boxes: Sorted list of bounding boxes
        texts: Sorted list of recognized texts
        y_threshold: Maximum y-distance to group texts into same line
        
    Returns:
        List of lines, each line is a list of {'text': str, 'box': list}
    """
    if not boxes or not texts:
        return []
    
    lines = []
    current_line = []
    current_y = None
    
    for box, text in zip(boxes, texts):
        # Get center y of current box
        y_coords = [point[1] for point in box]
        center_y = np.mean(y_coords)
        
        # Start new line if y distance is too large
        if current_y is None or abs(center_y - current_y) <= y_threshold:
            current_line.append({'text': text, 'box': box})
            if current_y is None:
                current_y = center_y
        else:
            # Save current line and start new one
            if current_line:
                lines.append(current_line)
            current_line = [{'text': text, 'box': box}]
            current_y = center_y
    
    # Add last line
    if current_line:
        lines.append(current_line)

    return lines


def format_ocr_as_paragraph(boxes: List[List], texts: List[str], y_threshold: float = 20) -> str:
    """
    Format OCR results as a structured paragraph with line breaks.
    
    Args:
        boxes: List of bounding boxes
        texts: List of recognized texts
        y_threshold: Maximum y-distance to group texts into same line
        
    Returns:
        Formatted text paragraph
    """
    # Sort by position
    sorted_boxes, sorted_texts = sort_boxes_by_position(boxes, texts)
    
    # Group into lines
    lines = group_text_by_lines(sorted_boxes, sorted_texts, y_threshold)
    
    # Format as paragraph
    paragraph_lines = []
    for line in lines:
        line_text = ' '.join([item['text'] for item in line])
        paragraph_lines.append(line_text)
    
    return '\n'.join(paragraph_lines)


def beautify_paragraph(paragraph: str) -> str:
    """
    Clean and normalize OCR-generated paragraph text to reduce OCR artifacts
    and improve readability for downstream LLM consumption.

    This function applies a sequence of heuristic fixes:
    - Normalize whitespace and punctuation spacing
    - Fix common OCR misspellings (small curated mapping)
    - Join digit groups broken by spaces (e.g., ID numbers, dates)
    - Normalize separators and remove repeated artifacts

    Args:
        paragraph: Raw paragraph text generated from OCR ordering

    Returns:
        Cleaned paragraph string
    """
    if not paragraph:
        return paragraph

    text = paragraph

    # Replace common OCR artifacts
    # remove control chars, weird unicode that often appears
    text = text.replace('\r', ' ')
    text = re.sub(r"\uFFFD", ' ', text)

    # Common manual corrections (can be extended)
    corrections = {
        'Independenice': 'Independence',
        'Hanpiness': 'Happiness',
        'Independenice _ Freedom _ Hanpiness': 'Independence - Freedom - Happiness',
        'sình': 'sinh',
        'Giới tinh': 'Giới tính',
        'NaM': 'Nam',
        'residece': 'residence',
        'Date ofaxpiry': 'Date of expiry',
        'Có giá tr] đến': 'Có giá trị đến',
        'ZOZ': '',
    }
    for k, v in corrections.items():
        text = text.replace(k, v)

    # Remove underscores and repeated non-letter artifacts, convert multiple -/_ to single dash
    text = re.sub(r'[_]{2,}', ' ', text)
    text = re.sub(r'[\u2012\u2013\u2014\-]{2,}', '-', text)
    text = re.sub(r'[\s]{2,}', ' ', text)

    # Normalize spacing around slashes, colons and commas
    text = re.sub(r'\s*/\s*', ' / ', text)
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'\s*,\s*', ', ', text)

    # Remove stray characters commonly introduced (e.g. stray brackets)
    text = re.sub(r'[\[\]\{\}]', '', text)

    # Join digit groups broken by spaces (e.g., '1 0/04/1988' -> '10/04/1988', '0750 880 04993' -> '075088004993')
    # Strategy: remove spaces between digits
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)

    # Fix dates with spaces around slashes (e.g., '10 / 04 / 1988') -> '10/04/1988'
    text = re.sub(r'\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})\b', r'\1/\2/\3', text)

    # Fix common OCR letter errors: repeated letters or near-duplicates
    # e.g., 'Independence  Freedom' -> 'Independence - Freedom' where underscores/dashes were intended
    text = re.sub(r'\s{2,}', ' ', text)

    # Trim and ensure consistent newlines: collapse multiple newlines to single
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = text.strip()

    # If a label and value are on the same line separated by multiple spaces, normalize to 'Label: value'
    # Example pattern: 'Họ và tên / Full name ĐINH NGỌC' -> 'Họ và tên / Full name: ĐINH NGỌC'
    def label_colonize(line: str) -> str:
        # If contains a bilingual label (contains '/' and then some uppercase words perhaps followed by value)
        if '/' in line:
            parts = line.split('/')
            # If last part contains both label and value, try to separate by detecting two consecutive uppercase words
            left = parts[0].strip()
            right = '/'.join(parts[1:]).strip()
            # if right has more than 3 words and contains uppercase then assume format 'Label Value...'
            words = right.split()
            if len(words) >= 2:
                # find first word that looks like a name/value: contains letters and at least one uppercase (heuristic)
                # If right already contains ':' then skip
                if ':' not in right:
                    # attempt to split label and value: find first token that starts with uppercase or is all-caps
                    for i in range(1, len(words)+1):
                        prefix = ' '.join(words[:i])
                        suffix = ' '.join(words[i:])
                        # heuristics: suffix should contain letters and not be empty
                        if suffix and re.search(r'[A-Za-zÀ-ỹ]', suffix):
                            # build normalized line
                            return f"{left} / {prefix}: {suffix}"
        return line

    lines = [label_colonize(ln.strip()) for ln in text.split('\n')]
    text = '\n'.join(lines)

    # Final trimming: remove duplicated spaces around punctuation
    text = re.sub(r'\s+([,.:;\-\\/])', r'\1', text)
    text = re.sub(r'([,.:;\-\\/])\s+', r'\1 ', text)

    return text


def create_ocr_context(ocr_results: Dict) -> str:
    """
    Create a structured text context from OCR results for LLM prompt.
    
    Args:
        ocr_results: OCR output from get_ocr_system
                     Structure: {
                       'results': [{
                         'filename': str,
                         'ocr_data': [{'text': str, 'confidence': float, 'box': list}],
                         'processing_time': {...}
                       }]
                     }
    
    Returns:
        Formatted text context for LLM
    """
    if not ocr_results or not ocr_results.get('results'):
        return ""
    
    context_parts = []
    
    for result in ocr_results['results']:
        filename = result.get('filename', 'unknown')
        ocr_data = result.get('ocr_data', [])
        
        if not ocr_data:
            continue
        
        # Extract boxes and texts
        boxes = [item['box'] for item in ocr_data]
        texts = [item['text'] for item in ocr_data]
        
        # Format as paragraph
        paragraph = format_ocr_as_paragraph(boxes, texts)
        # Beautify paragraph to fix OCR artifacts before sending to LLM
        paragraph = beautify_paragraph(paragraph)
        print(f"paragraph: {paragraph}")
        
        context_parts.append(f"=== Image: {filename} ===\n{paragraph}")
    
    return "\n\n".join(context_parts)


def process_ocr_images(files: List, display_text_box: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a list of image files through OCR pipeline.
    
    Args:
        files: List of file objects from Flask request
        display_text_box: If True, include ocr_data in results; if False, exclude it
        
    Returns:
        Tuple of (results, errors)
        - results: List of OCR results for successful images
        - errors: List of errors encountered
    """
    results = []
    errors = []
    
    try:
        ocr_system = get_ocr_system()
    except Exception as e:
        return results, [{
            'file_index': -1,
            'filename': 'global',
            'error': f'OCR system not initialized: {str(e)}'
        }]
    
    for idx, file in enumerate(files):
        try:
            # Validate file
            if file.filename == '':
                errors.append({
                    'file_index': idx,
                    'filename': 'unknown',
                    'error': 'Empty filename'
                })
                continue
            
            # Preprocess image
            image = preprocess_image(file)
            
            # Run OCR
            ocr_res, time_det, time_rec = ocr_system.ocr(image, det=True, rec=True)
            
            # Format results
            file_result = {
                'filename': file.filename,
                'processing_time': {
                    'detection_ms': round(time_det * 1000, 2),
                    'recognition_ms': round(time_rec * 1000, 2)
                }
            }
            
            # Only add ocr_data if display_text_box is True
            if display_text_box:
                file_result['ocr_data'] = []
                
                if ocr_res and len(ocr_res) > 0:
                    for box, rec_result in ocr_res[0]:
                        text, confidence = rec_result
                        if text and text.strip():  # Only include non-empty results
                            file_result['ocr_data'].append({
                                'text': text,
                                'confidence': round(float(confidence), 4),
                                'box': box
                            })
            else:
                # Store ocr_data temporarily for context creation, but don't include in final result
                if ocr_res and len(ocr_res) > 0:
                    temp_ocr_data = []
                    for box, rec_result in ocr_res[0]:
                        text, confidence = rec_result
                        if text and text.strip():
                            temp_ocr_data.append({
                                'text': text,
                                'confidence': round(float(confidence), 4),
                                'box': box
                            })
                    file_result['_temp_ocr_data'] = temp_ocr_data
            
            results.append(file_result)
            
        except ValueError as e:
            errors.append({
                'file_index': idx,
                'filename': file.filename,
                'error': str(e)
            })
        except Exception as e:
            errors.append({
                'file_index': idx,
                'filename': file.filename,
                'error': f'Processing error: {str(e)}'
            })
    
    return results, errors


def process_extract_images(files: List, info_dict: Dict[str, str], 
                          use_parallel: bool = True,
                          batch_size: int = 3,
                          llm_timeout: int = 30,
                          display_text_box: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Process images through full OCR + LLM extraction pipeline.
    
    Args:
        files: List of file objects from Flask request
        info_dict: Dict mapping field_key -> field_description (Vietnamese description)
        use_parallel: Use parallel LLM extraction
        batch_size: Batch size for parallel extraction
        llm_timeout: Timeout for LLM queries in seconds
        display_text_box: If True, include ocr_data in results; if False, exclude it
        
    Returns:
        Tuple of (results, errors)
        - results: List of extraction results with extracted_fields
        - errors: List of errors encountered
    """
    results = []
    errors = []
    
    # First, process all images through OCR
    ocr_results, ocr_errors = process_ocr_images(files, display_text_box=True)
    errors.extend(ocr_errors)
    
    # Format OCR results for context creation
    ocr_data = {
        'results': ocr_results
    }
    
    # Process each OCR result with LLM extraction
    for idx, ocr_item in enumerate(ocr_results):
        try:
            # Create context from OCR results
            single_result_dict = {'results': [ocr_item]}
            context = create_ocr_context(single_result_dict)
            
            if not context or not context.strip():
                errors.append({
                    'file_index': idx,
                    'filename': ocr_item['filename'],
                    'error': 'No text detected in image'
                })
                continue
            
            # Extract fields using LLM with descriptions
            if use_parallel:
                from app.vision.ocr.llm_extractor import parallel_field_extraction_with_descriptions
                extracted_fields = parallel_field_extraction_with_descriptions(
                    context,
                    info_dict,
                    batch_size=batch_size,
                    timeout=llm_timeout
                )
            else:
                # Single-threaded extraction
                from app.vision.ocr.llm_extractor import LLMClient
                client = LLMClient()
                extracted_fields = {}
                for field_key, description in info_dict.items():
                    user_message = f"QUERY: Từ đoạn text dưới đây, trích xuất trường thông tin sau: {description}\n\nINSTRUCTION: Trả về dạng JSON với key là '{field_key}', không giải thích gì thêm.\nCONTEXT: {context}"
                    response = client.query(user_message, timeout=llm_timeout)
                    
                    if response:
                        try:
                            response_str = response.strip()
                            if response_str.startswith('```json'):
                                response_str = response_str[7:]
                            if response_str.startswith('```'):
                                response_str = response_str[3:]
                            if response_str.endswith('```'):
                                response_str = response_str[:-3]
                            response_str = response_str.strip()
                            
                            parsed = json.loads(response_str)
                            extracted_fields[field_key] = parsed.get(field_key, "")
                        except (json.JSONDecodeError, AttributeError, TypeError) as e:
                            print(f"Failed to parse response for field {field_key}: {e}")
                            extracted_fields[field_key] = response or ""
                    else:
                        extracted_fields[field_key] = ""
            
            # Format result
            file_result = {
                'filename': ocr_item['filename'],
                'extracted_fields': extracted_fields,
                'processing_time': ocr_item['processing_time']
            }
            
            # Only add ocr_data if display_text_box is True
            if display_text_box:
                file_result['ocr_data'] = ocr_item.get('ocr_data', [])
            
            results.append(file_result)
            
        except Exception as e:
            errors.append({
                'file_index': idx,
                'filename': ocr_item['filename'],
                'error': f'Extraction error: {str(e)}'
            })
    
    return results, errors
