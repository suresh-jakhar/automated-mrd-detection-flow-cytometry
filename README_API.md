# MRD Detection FastAPI Deployment

Complete FastAPI application for MRD detection using a trained VAE model.

## 📋 Overview

- **Model**: Variational Autoencoder (VAE) with 14 input features → 4-dimensional latent space
- **Input**: One or more CSV files (same scan) with per-cell feature vectors
- **Output**: MRD% (percentage of cells with reconstruction error above threshold)
- **Threshold**: 0.0231 (calibrated from training on 6 patient samples)

## 🚀 Quick Start

### Local Development (CPU)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**:
   ```bash
   python app.py
   ```
   or
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the API**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

4. **Test the API**:
   ```bash
   python test_client.py
   ```

### Docker Deployment

1. **Build image**:
   ```bash
   docker build -t mrd-detection-api:latest .
   ```

2. **Run container**:
   ```bash
   docker run -p 8000:8000 mrd-detection-api:latest
   ```

3. **Or use Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Check logs**:
   ```bash
   docker-compose logs -f mrd-api
   ```

## 📝 API Endpoints

### POST /predict
**Predict MRD% from one or more CSV files**

**Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "files=@file1.csv" \
  -F "files=@file2.csv" \
  -F "files=@file3.csv" \
  -F "return_details=true"
```

**Response**:
```json
{
  "mrd_pct": 3.45,
  "abnormal_count": 1234,
  "total": 35680,
  "threshold": 0.02310833333333333,
  "mean_error": 0.0156,
  "std_error": 0.0089,
  "min_error": 0.0001,
  "max_error": 0.0512,
  "num_files": 3,
  "file_names": ["file1.csv", "file2.csv", "file3.csv"],
  "processing_time_seconds": 2.345,
  "device": "cpu"
}
```

**Query Parameters**:
- `threshold` (float, optional): Override default threshold (0.0231)
- `return_details` (bool, default: false): Include detailed stats

### GET /health
**Health check endpoint**

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": 1705227600.123
}
```

### GET /info
**Model information**

```bash
curl http://localhost:8000/info
```

Response:
```json
{
  "model_name": "Variational Autoencoder (VAE) for MRD Detection",
  "input_dim": 14,
  "latent_dim": 4,
  "threshold": 0.02310833333333333,
  "device": "cpu",
  "batch_size": 4096,
  "architecture": "14 -> 16 -> 8 -> 4 (latent) -> 4 -> 8 -> 16 -> 14"
}
```

### POST /predict-batch
**Advanced batch prediction with optional label validation**

Useful for evaluating the model on labeled datasets.

## 📊 Input Data Format

CSV files should contain:
- **14 numerical feature columns** (feature values, already scaled or will be normalized by MinMaxScaler)
- **Optional 'Time' column** (will be dropped automatically)
- **No header requirement** (pandas assumes header by default; adjust as needed)

Example:
```csv
feature_0,feature_1,feature_2,...,feature_13
0.123,0.456,0.789,...,0.234
0.111,0.222,0.333,...,0.444
...
```

## ⚙️ Configuration

Edit `app.py` to modify:

```python
# Model configuration
MODEL_PATH = "model/vae_4dim_6_final.pth"
INPUT_DIM = 14
LATENT_DIM = 4
THRESHOLD = 0.02310833333333333
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4096
```

## 🔧 Performance Notes

- **Batch size**: 4096 cells per batch (adjust BATCH_SIZE for memory/speed tradeoff)
- **Device**: Automatically uses GPU (CUDA) if available, falls back to CPU
- **Large files**: Processes streaming/chunked to avoid OOM on millions of cells
- **Expected throughput** (CPU): ~10k cells/second

## 📦 File Structure

```
.
├── app.py                        # FastAPI application
├── model_wrapper.py              # VAE model class
├── inference.py                  # Inference & preprocessing logic
├── test_client.py                # Sample client for testing
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Docker Compose configuration
├── model/
│   └── vae_4dim_6_final.pth     # Trained model checkpoint
└── README.md                     # This file
```

## 🧪 Testing

### Local Testing
```bash
# Run sample client
python test_client.py

# Manual curl test
curl -X POST "http://localhost:8000/predict" \
  -F "files=@sample_data_1.csv" \
  -F "return_details=true"
```

### Load Testing (with locust)
```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```

## 📈 MRD% Interpretation

- **MRD% < 1%**: Likely healthy/negative
- **MRD% 1-3%**: Borderline/monitoring needed
- **MRD% > 3%**: Elevated risk (calibration dependent)

*Note: Thresholds are calibrated on 6 patient samples from training. Validate with your own labeled data.*

## 🐛 Troubleshooting

**Model not loading**:
- Check `model/vae_4dim_6_final.pth` exists
- Verify PyTorch version compatibility

**Dimension mismatch**:
- Ensure CSV has exactly 14 features (excluding 'Time' column)
- Remove any extra columns before uploading

**Memory issues with large files**:
- Reduce BATCH_SIZE in app.py
- Process files in chunks before uploading

**Slow inference**:
- Use GPU if available (set `DEVICE = 'cuda'`)
- Increase batch size for better throughput

## 📞 Support & Contribution

For issues or improvements, refer to the training notebook (`model/training.ipynb`) for original model details and calibration steps.

---

**Last Updated**: January 2026
**Model Version**: vae_4dim_6_final
**PyTorch Version**: 2.9.1
