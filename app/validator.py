from functools import wraps
from flask import request, jsonify
from app.extentions import settings


def require_token(f):
    """Decorator to validate hard token from request."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not settings.HARD_TOKEN:
            # Token validation disabled
            return f(*args, **kwargs)
        
        token = request.form.get('hard_token') or request.headers.get('X-API-Token')
        
        if token is None or token != settings.HARD_TOKEN:
            return jsonify(
                success=False,
                message='Invalid or missing authentication token',
                error_code='UNAUTHORIZED'
            ), 401
        
        return f(*args, **kwargs)
    
    return decorated_function
