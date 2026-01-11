import torch

class ModelWrapper:
    """Model loader that defers loading until needed and provides helpful APIs.

    - Loading is lazy by default to avoid crashing import-time in environments where the model
      file may be a state_dict or otherwise not directly loadable.
    - Callers may call `set_model(model)` to inject a ready model instance (useful for tests).
    """
    def __init__(self, path, device='cpu', load_on_init=False):
        self.path = path
        self.device = device
        self.model = None
        self._loaded = False
        self._load_error = None
        if load_on_init:
            try:
                self._load()
            except Exception as e:
                self._load_error = e

    def _load(self):
        obj = torch.load(self.path, map_location=self.device)
        import torch.nn as nn
        if isinstance(obj, nn.Module):
            self.model = obj.eval().to(self.device)
            self._loaded = True
            return

        # If file contains state_dict/parameters, provide clear guidance
        if isinstance(obj, dict):
            if 'state_dict' in obj or any(isinstance(k, str) for k in obj.keys()):
                raise RuntimeError(
                    "File contains state_dict/parameters; ModelWrapper cannot instantiate the model class. "
                    "Save a full model object with `torch.save(model)` or provide your model class and load the state_dict."
                )

        raise RuntimeError(f"Unsupported model object in {self.path} (type={type(obj)}). Save model object with torch.save(model).")

    def _ensure_loaded(self):
        if self._loaded and self.model is not None:
            return
        try:
            self._load()
        except Exception as e:
            self._load_error = e
            raise RuntimeError(f"Model load failed: {e}") from e

    def set_model(self, model):
        """Inject a ready-to-use model instance (useful for tests or custom loading)."""
        self.model = model
        self._loaded = True
        self._load_error = None

    def is_loaded(self):
        return self._loaded and self.model is not None

    def predict(self, x_tensor):
        # Lazy load on first inference attempt
        if not self.is_loaded():
            self._ensure_loaded()

        if self.model is None:
            raise RuntimeError("Model not loaded as nn.Module. See initialization error.")

        self.model.eval()
        x = x_tensor.to(self.device)
        with torch.no_grad():
            out = self.model(x)

        # Normalize output: return torch.Tensor when available, otherwise Python native
        if isinstance(out, torch.Tensor):
            return out.detach().cpu()
        return out
