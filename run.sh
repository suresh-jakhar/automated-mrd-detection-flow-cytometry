#!/bin/bash
# Quick start script for MRD Detection API (Linux/macOS)

set -e

echo "=========================================="
echo "MRD Detection API - Quick Start"
echo "=========================================="

# Check if model file exists
if [ ! -f "model/vae_4dim_6_final.pth" ]; then
    echo "❌ Model file not found: model/vae_4dim_6_final.pth"
    exit 1
fi

echo "✓ Model file found"

# Create Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To start the API, run:"
echo "  uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "API will be available at:"
echo "  - http://localhost:8000"
echo "  - API docs: http://localhost:8000/docs"
echo ""
echo "To test the API, run in another terminal:"
echo "  python test_client.py"
echo ""
