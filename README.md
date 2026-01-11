# automated-mrd-detection-flow-cytometry

#test

## Run the Flask API 🚀

1. Create a Python virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install required packages:

```powershell
pip install -r requirements.txt
```

3. Start the server:

```powershell
python app.py
```

The API will listen on `http://0.0.0.0:5000`.

## Example request ✉️

POST JSON to `/predict` with key `input` (list or nested lists). Example using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"input": [[0.1, 0.2, 0.3, 0.4]]}' \
  http://127.0.0.1:5000/predict
```

Or run the sample client:

```powershell
python example_client.py
```

## Notes / Troubleshooting ⚠️

- The loader expects the `.pth` to contain a pickled `nn.Module` object (save with `torch.save(model)`).
- If the file contains only a `state_dict`, you must provide the model class and load the state dict yourself.
- If you get import errors for `flask`/`torch` in your editor, install requirements listed above.
