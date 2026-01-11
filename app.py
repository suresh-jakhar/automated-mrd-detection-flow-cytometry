import os
import logging
from flask import Flask, request, jsonify
from model_loader import ModelWrapper
import torch

app = Flask(__name__)

# Basic production-safe defaults
app.config.setdefault('MAX_CONTENT_LENGTH', 1 * 1024 * 1024)  # 1 MB
app.config.setdefault('JSONIFY_PRETTYPRINT_REGULAR', False)

# Operational limits (tunable via environment)
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 64))
MAX_FEATURES = int(os.environ.get('MAX_FEATURES', 1024))

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Path to the model file in the repository
MODEL_PATH = "model/vae_4dim_6_final.pth"

# Instantiate the model wrapper but defer loading until needed (safer at import-time)
model = ModelWrapper(MODEL_PATH, load_on_init=False)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/ready', methods=['GET'])
def ready():
    # Provide basic readiness info (no sensitive details)
    is_ready = bool(getattr(model, 'is_loaded', lambda: False)())
    load_error = getattr(model, '_load_error', None)
    return jsonify({'ready': is_ready,
                    'load_error': None if is_ready else (str(load_error) if load_error else '')})


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({'error': 'Request payload too large'}), 413


@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': getattr(e, 'description', 'Bad request')}), 400


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    # Log exception details server-side but don't expose them to callers
    app.logger.exception('Unhandled exception during request handling')
    return jsonify({'error': 'Internal server error'}), 500


def validate_input_payload(data):
    import numpy as _np
    if not data or 'input' not in data:
        raise ValueError('JSON must contain the key "input"')
    try:
        arr = _np.array(data['input'], dtype=float)
    except Exception as exc:
        raise ValueError(f'Invalid input format: {exc}')

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Enforce sensible limits
    if arr.shape[0] > MAX_BATCH_SIZE:
        raise ValueError(f'Batch size {arr.shape[0]} exceeds max {MAX_BATCH_SIZE}')
    if arr.shape[1] > MAX_FEATURES:
        raise ValueError(f'Number of features {arr.shape[1]} exceeds max {MAX_FEATURES}')

    return arr


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure Content-Type and JSON parseability
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    # Quick check for oversized payloads to ensure consistent 413 behavior
    content_len = request.content_length
    max_len = app.config.get('MAX_CONTENT_LENGTH')
    if content_len is not None and max_len is not None and content_len > max_len:
        return jsonify({'error': 'Request payload too large'}), 413

    try:
        data = request.get_json()
    except Exception:
        return jsonify({'error': 'Invalid JSON'}), 400

    try:
        arr = validate_input_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    tensor = torch.from_numpy(arr).float()

    # Run inference with robust error handling
    try:
        out = model.predict(tensor)
    except RuntimeError as e:
        # Differentiate between model not ready vs other runtime errors
        msg = str(e)
        app.logger.error('Model runtime error: %s', msg)
        if 'not loaded' in msg or 'Model load failed' in msg:
            return jsonify({'error': 'Model not ready', 'details': msg}), 503
        return jsonify({'error': f'Model inference error: {msg}'}), 500
    except Exception as e:
        app.logger.exception('Unexpected error during model inference')
        return jsonify({'error': 'Model inference error'}), 500

    # Convert output to native lists
    if isinstance(out, torch.Tensor):
        out_list = out.detach().cpu().tolist()
    else:
        out_list = out

    return jsonify({'prediction': out_list})


if __name__ == '__main__':
    # For production, prefer a WSGI server (gunicorn/uvicorn) and ensure DEBUG is False
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
