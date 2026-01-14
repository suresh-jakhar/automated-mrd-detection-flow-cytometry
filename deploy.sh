#!/bin/bash
# Deployment script for production

set -e

echo "=========================================="
echo "MRD Detection API - Production Deployment"
echo "=========================================="

# Configuration
ENVIRONMENT=${1:-production}
IMAGE_TAG=${2:-latest}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"your-registry"}

echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"

# Step 1: Build Docker image
echo ""
echo "Step 1: Building Docker image..."
docker build -t $DOCKER_REGISTRY/mrd-detection-api:$IMAGE_TAG .

# Step 2: Run tests in container
echo ""
echo "Step 2: Running tests..."
docker run --rm $DOCKER_REGISTRY/mrd-detection-api:$IMAGE_TAG \
    bash -c "pip install -r requirements-dev.txt && pytest tests/ -v"

# Step 3: Push to registry
echo ""
echo "Step 3: Pushing to registry..."
docker push $DOCKER_REGISTRY/mrd-detection-api:$IMAGE_TAG

# Step 4: Deploy based on environment
echo ""
echo "Step 4: Deploying to $ENVIRONMENT..."

if [ "$ENVIRONMENT" = "production" ]; then
    # Production deployment (example with kubectl)
    kubectl set image deployment/mrd-api \
        mrd-api=$DOCKER_REGISTRY/mrd-detection-api:$IMAGE_TAG \
        --namespace=production
    
    kubectl rollout status deployment/mrd-api --namespace=production
    
elif [ "$ENVIRONMENT" = "staging" ]; then
    # Staging deployment
    kubectl set image deployment/mrd-api \
        mrd-api=$DOCKER_REGISTRY/mrd-detection-api:$IMAGE_TAG \
        --namespace=staging
    
    kubectl rollout status deployment/mrd-api --namespace=staging
    
else
    echo "Unknown environment: $ENVIRONMENT"
    exit 1
fi

# Step 5: Run smoke tests
echo ""
echo "Step 5: Running smoke tests..."
sleep 10  # Wait for deployment to stabilize

if [ "$ENVIRONMENT" = "production" ]; then
    export API_URL="https://api.production.example.com"
else
    export API_URL="https://api.staging.example.com"
fi

python -m pytest tests/smoke/ -v

echo ""
echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo "Environment: $ENVIRONMENT"
echo "Image: $DOCKER_REGISTRY/mrd-detection-api:$IMAGE_TAG"
echo "URL: $API_URL"
echo "=========================================="
