# CI/CD and Deployment Guide

## Overview

This project includes comprehensive CI/CD pipelines and automated testing for production-ready deployment.

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

**CI Pipeline** (`.github/workflows/ci.yml`):
- Runs on push/PR to `main` and `develop` branches
- Linting with flake8 and black
- Unit tests with pytest
- Integration tests
- Code coverage reporting
- Docker image build and test
- Security scanning with Trivy

**CD Pipeline** (`.github/workflows/cd.yml`):
- Deploys to staging on push to `main`
- Deploys to production on version tags (`v*`)
- Manual deployment trigger available
- Smoke tests after deployment
- Slack notifications

### Required Secrets

Configure these in GitHub Settings → Secrets:

```
DOCKER_USERNAME         # Docker Hub username
DOCKER_PASSWORD         # Docker Hub password/token
STAGING_URL            # Staging environment URL
PRODUCTION_URL         # Production environment URL
SLACK_WEBHOOK          # Slack webhook for notifications
```

## 🧪 Automated Testing

### Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_model.py      # Model tests
│   └── test_inference.py  # Inference logic tests
├── integration/           # Integration tests
│   └── test_api.py        # API endpoint tests
└── smoke/                 # Smoke tests
    └── test_smoke.py      # Quick validation tests
```

### Running Tests Locally

**Install test dependencies**:
```bash
pip install -r requirements-dev.txt
```

**Run all tests**:
```bash
pytest tests/ -v
```

**Run specific test types**:
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Smoke tests only
pytest tests/smoke/ -v -m smoke

# With coverage
pytest tests/ --cov=. --cov-report=html
```

**Run fast tests during development**:
```bash
pytest tests/unit/ -v --maxfail=1 --tb=short
```

## 📦 Deployment Options

### Option 1: Docker Compose (Development/Testing)

```bash
docker-compose up -d
```

### Option 2: Kubernetes (Production)

**Deploy to production**:
```bash
kubectl apply -f k8s/production/deployment.yaml
```

**Check deployment status**:
```bash
kubectl rollout status deployment/mrd-detection-api -n production
```

**View logs**:
```bash
kubectl logs -f deployment/mrd-detection-api -n production
```

### Option 3: Using Deployment Script

```bash
# Deploy to staging
bash deploy.sh staging v1.0.0

# Deploy to production
bash deploy.sh production v1.0.0
```

## 🚀 Release Process

### Creating a New Release

1. **Update version** in relevant files
2. **Run tests locally**:
   ```bash
   pytest tests/ -v
   ```
3. **Commit changes**:
   ```bash
   git add .
   git commit -m "Release v1.0.0"
   ```
4. **Create and push tag**:
   ```bash
   git tag v1.0.0
   git push origin main --tags
   ```
5. **GitHub Actions will automatically**:
   - Run all tests
   - Build Docker image
   - Push to registry
   - Deploy to production
   - Run smoke tests
   - Create GitHub release

## 🔍 Monitoring

### Health Checks

**Kubernetes liveness probe**:
- Endpoint: `/health`
- Initial delay: 30s
- Period: 10s

**Kubernetes readiness probe**:
- Endpoint: `/health`
- Initial delay: 10s
- Period: 5s

### Manual Health Check

```bash
# Check health
curl http://your-api-url/health

# Check model info
curl http://your-api-url/info
```

## 📊 Code Coverage

After running tests with coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

View the report:
```bash
open htmlcov/index.html
```

## 🛡️ Security

### Dependency Scanning

Automatically runs on CI pipeline using Trivy.

**Manual scan**:
```bash
pip install safety
safety check -r requirements.txt
```

### Container Security

**Scan Docker image**:
```bash
docker scan mrd-detection-api:latest
```

## 🔧 Troubleshooting

### Tests Failing Locally

1. **Ensure all dependencies installed**:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

2. **Check model file exists**:
   ```bash
   ls -lh model/vae_4dim_6_final.pth
   ```

3. **Clear pytest cache**:
   ```bash
   pytest --cache-clear
   ```

### CI Pipeline Failures

1. **Check GitHub Actions logs**
2. **Verify secrets are configured**
3. **Run tests locally to reproduce**

### Deployment Issues

1. **Check pod status**:
   ```bash
   kubectl get pods -n production
   ```

2. **View pod logs**:
   ```bash
   kubectl logs <pod-name> -n production
   ```

3. **Describe pod for events**:
   ```bash
   kubectl describe pod <pod-name> -n production
   ```

## 📝 Best Practices

1. **Always run tests before pushing**:
   ```bash
   pytest tests/ -v
   ```

2. **Use feature branches**:
   ```bash
   git checkout -b feature/new-feature
   ```

3. **Keep dependencies updated**:
   ```bash
   pip list --outdated
   ```

4. **Monitor test coverage**:
   - Aim for >80% coverage
   - Focus on critical paths

5. **Use semantic versioning**:
   - Major: Breaking changes (v2.0.0)
   - Minor: New features (v1.1.0)
   - Patch: Bug fixes (v1.0.1)

## 🎯 Continuous Improvement

### Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures for setup
4. Add appropriate markers (`@pytest.mark.unit`, etc.)

### Updating CI/CD

1. Modify `.github/workflows/*.yml`
2. Test changes on feature branch first
3. Monitor workflow runs after merge

### Performance Testing

Consider adding:
- Load testing with Locust
- Performance benchmarks
- Memory profiling

---

**Status**: ✅ Production Ready with Full CI/CD  
**Last Updated**: January 2026
