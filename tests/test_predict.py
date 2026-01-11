import model_loader
# Prevent the real model from being loaded at import time during tests
class SilentModelWrapper:
    def __init__(self, path, **kwargs):
        # Do not load anything from disk; accept kwargs for compatibility with the real API
        self.path = path
        self._loaded = False
        self._load_error = None
    def predict(self, x):
        batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
        return [[1.0, 2.0] for _ in range(batch_size)]
model_loader.ModelWrapper = SilentModelWrapper

from app import app
import torch

class DummyModel:
    def eval(self):
        return self
    def to(self, device):
        return self
    def __call__(self, x):
        # Return a native Python list so we don't require torch in the test env
        batch_size = len(x)
        return [[1.0, 2.0] for _ in range(batch_size)]
    def predict(self, x):
        return self.__call__(x)

class DummyModelTensor:
    def predict(self, x):
        batch_size = x.shape[0]
        return torch.tensor([[0.1, 0.2] for _ in range(batch_size)])


def test_predict_success(monkeypatch):
    # Replace the model instance used by the app with a dummy
    monkeypatch.setattr('app.model', DummyModel())
    client = app.test_client()
    resp = client.post('/predict', json={'input': [[1, 2, 3, 4]]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], list)


def test_missing_json_returns_400():
    client = app.test_client()
    # Send non-JSON body
    resp = client.post('/predict', data='not json', content_type='text/plain')
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert 'error' in data
    assert 'Request must be JSON' in data['error']


def test_missing_input_key_returns_400():
    client = app.test_client()
    resp = client.post('/predict', json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert 'error' in data
    assert 'must contain the key "input"' in data['error']


def test_invalid_input_format_returns_400():
    client = app.test_client()
    # Pass a value that cannot be converted to float
    resp = client.post('/predict', json={'input': ['not convertible']})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert 'error' in data
    assert 'Invalid input format' in data['error']


def test_model_inference_error_returns_500(monkeypatch):
    # Monkeypatch the model's predict to raise an exception
    class BrokenModel:
        def predict(self, x):
            raise RuntimeError('boom')
    monkeypatch.setattr('app.model', BrokenModel())
    client = app.test_client()
    resp = client.post('/predict', json={'input': [[1,2,3,4]]})
    assert resp.status_code == 500
    data = resp.get_json()
    assert data is not None
    assert 'error' in data
    assert 'Model inference error' in data['error']


def test_predict_handles_tensor_output(monkeypatch):
    # Monkeypatch to a model that returns a torch.Tensor
    monkeypatch.setattr('app.model', DummyModelTensor())
    client = app.test_client()
    resp = client.post('/predict', json={'input': [[1,2,3,4]]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], list)
    assert isinstance(data['prediction'][0], list)


def test_health_and_ready_endpoints():
    client = app.test_client()
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_json().get('status') == 'ok'

    r2 = client.get('/ready')
    assert r2.status_code == 200
    data = r2.get_json()
    assert 'ready' in data


def test_malformed_json_returns_400():
    client = app.test_client()
    resp = client.post('/predict', data='[invalid', content_type='application/json')
    assert resp.status_code == 400
    assert 'Invalid JSON' in resp.get_json().get('error')


def test_oversized_batch_returns_400():
    client = app.test_client()
    large_batch = [[1,2,3]] * 100  # default MAX_BATCH_SIZE is 64
    resp = client.post('/predict', json={'input': large_batch})
    assert resp.status_code == 400
    assert 'exceeds max' in resp.get_json().get('error')


def test_request_too_large(monkeypatch):
    # Reduce limit for the test so we don't actually need >1MB payload
    monkeypatch.setitem(app.config, 'MAX_CONTENT_LENGTH', 10)
    data = 'x' * 200
    resp = app.test_client().post('/predict', data=data, content_type='application/json')
    assert resp.status_code == 413


def test_model_not_ready_returns_503(monkeypatch):
    class LazyModel:
        def predict(self, x):
            raise RuntimeError("Model not loaded")
    monkeypatch.setattr('app.model', LazyModel())
    resp = app.test_client().post('/predict', json={'input': [[1,2,3,4]]})
    assert resp.status_code == 503
    assert 'Model not ready' in resp.get_json().get('error')


def test_debug_off_by_default():
    assert not app.debug
