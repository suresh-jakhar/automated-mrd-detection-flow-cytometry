"""
FastAPI application for MRD detection.
Handles file uploads, preprocessing, batch inference, and MRD% calculation.
"""

import os
import io
import time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch
from inference import MRDPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MRD Detection API",
    description="API for predicting MRD% from flow cytometry data using VAE",
    version="1.0.0",
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup)
predictor = None

# Configuration - allow override from environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "model/vae_4dim_6_final.pth")
SCALER_PATH = None  # Set if you have a saved scaler
INPUT_DIM = 14
LATENT_DIM = 4
THRESHOLD = 0.02310833333333333
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096


@app.on_event("startup")
def load_model():
    """Load model on app startup."""
    global predictor
    try:
        predictor = MRDPredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            input_dim=INPUT_DIM,
            latent_dim=LATENT_DIM,
            threshold=THRESHOLD,
            device=DEVICE,
        )
        logger.info(f"Model loaded successfully on device: {DEVICE}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "device": DEVICE,
        "timestamp": time.time(),
    }


@app.get("/info")
def model_info():
    """Get model information."""
    return {
        "model_name": "Variational Autoencoder (VAE) for MRD Detection",
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "threshold": THRESHOLD,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "architecture": "14 -> 16 -> 8 -> 4 (latent) -> 4 -> 8 -> 16 -> 14",
    }


@app.post("/predict")
async def predict_mrd(
    files: List[UploadFile] = File(...),
    threshold: Optional[float] = Query(
        None, description="Optional override for reconstruction error threshold"
    ),
    return_details: bool = Query(False, description="Return detailed statistics"),
):
    """
    Predict MRD% from uploaded flow cytometry CSV file(s).

    Multiple files will be concatenated (same scan, e.g., 2-3 files per scan).
    Computes MRD% as percentage of cells with reconstruction error > threshold.

    Args:
        files: One or more CSV files containing per-cell feature vectors (14 features each)
        threshold: Optional override for the reconstruction error threshold
        return_details: If True, return detailed stats

    Returns:
        JSON with MRD%, abnormal cell count, total cell count, and optional detailed stats
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        start_time = time.time()

        # Read and concatenate all uploaded files
        dataframes = []
        total_rows = 0

        for file in files:
            # Check file extension
            if not file.filename.endswith(".csv"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file format: {file.filename}. Only CSV files are supported.",
                )

            # Read CSV
            try:
                content = await file.read()
                df = pd.read_csv(io.StringIO(content.decode("utf-8")))

                # Drop 'Time' column if present (matching training pipeline)
                if "Time" in df.columns:
                    df = df.drop(columns=["Time"])

                dataframes.append(df)
                total_rows += len(df)
                logger.info(
                    f"Loaded {file.filename}: {len(df)} rows, {df.shape[1]} features"
                )

            except pd.errors.ParserError as e:
                raise HTTPException(
                    status_code=400, detail=f"Error parsing {file.filename}: {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Error reading {file.filename}: {str(e)}"
                )

        # Concatenate all dataframes
        if not dataframes:
            raise HTTPException(status_code=400, detail="No valid data found in files")

        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined data shape: {combined_df.shape}")

        # Validate feature count
        if combined_df.shape[1] != INPUT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {INPUT_DIM} features, but got {combined_df.shape[1]}. "
                f"Ensure Time column is removed and all features are present.",
            )

        # Convert to numpy and preprocess
        X = combined_df.values
        X_preprocessed = predictor.preprocess(X)

        # Predict MRD
        if return_details:
            result = predictor.predict_mrd(
                X_preprocessed,
                batch_size=BATCH_SIZE,
                threshold=threshold,
                return_details=True,
            )
        else:
            mrd_pct = predictor.predict_mrd(
                X_preprocessed,
                batch_size=BATCH_SIZE,
                threshold=threshold,
                return_details=False,
            )
            result = {"mrd_pct": mrd_pct}

        # Add metadata
        result["num_files"] = len(files)
        result["file_names"] = [f.filename for f in files]
        result["processing_time_seconds"] = round(time.time() - start_time, 3)
        result["device"] = DEVICE

        logger.info(
            f"Prediction completed in {result['processing_time_seconds']}s. MRD%: {result.get('mrd_pct', 'N/A')}"
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict-batch")
async def predict_batch_with_labels(
    file: UploadFile = File(...),
    label_column: str = Query(
        "mrd_label", description="Column name for true MRD% labels (optional)"
    ),
):
    """
    Advanced endpoint for batch predictions with optional label validation.
    Useful for evaluation/calibration on labeled datasets.

    Expects CSV with feature columns + optional label column.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        # Drop Time if present
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        # Separate features and labels if label column exists
        has_labels = label_column in df.columns
        if has_labels:
            labels = df[label_column].values
            features_df = df.drop(columns=[label_column])
        else:
            labels = None
            features_df = df

        # Validate feature count
        if features_df.shape[1] != INPUT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {INPUT_DIM} features, got {features_df.shape[1]}",
            )

        # Preprocess and predict
        X = features_df.values
        X_preprocessed = predictor.preprocess(X)
        result = predictor.predict_mrd(
            X_preprocessed, batch_size=BATCH_SIZE, return_details=True
        )

        # Add label comparison if available
        if has_labels:
            result["label_column"] = label_column
            result["mean_label"] = float(np.mean(labels))
            result["std_label"] = float(np.std(labels))
            result["error_vs_label"] = float(abs(result["mrd_pct"] - np.mean(labels)))

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Root endpoint with API documentation."""
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
