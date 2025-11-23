# Docker Deployment Guide for Azure

Complete guide to deploy PPAnalyzer backend using Docker containers.

---

## 🐳 Why Docker?

- ✅ Solves GLIBC compatibility issues
- ✅ Works with any Python version
- ✅ Perfect for ML models (PyTorch, transformers)
- ✅ Consistent across environments
- ✅ Industry standard

---

## 📋 Prerequisites

- Docker Desktop installed (`brew install --cask docker`)
- Azure CLI installed and logged in
- Code pushed to GitHub

---

## 🚀 Deployment Steps

### **Step 1: Test Docker Locally (Optional)**

```bash
cd /Users/saeedalketbi/Desktop/Senior\ design\ 2/project/ppaudit

# Build Docker image
cd backend
docker build -t ppanalyzer-backend .

# Test locally
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your_key" \
  -e HEYGEN_API_KEY="your_key" \
  ppanalyzer-backend

# Test in browser: http://localhost:8000/api/health
```

---

### **Step 2: Create Azure Container Registry**

```bash
# Set variables
RESOURCE_GROUP="ppanalyzer-rg"
REGISTRY_NAME="ppanalyzerregistry"  # Must be globally unique, lowercase only
LOCATION="uaenorth"

# Create container registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --sku Basic \
  --location $LOCATION \
  --admin-enabled true

echo "✓ Container Registry created: $REGISTRY_NAME.azurecr.io"
```

---

### **Step 3: Configure Web App for Docker**

```bash
BACKEND_NAME="ppanalyzer-backend"

# Delete existing web app (created without Docker)
az webapp delete --resource-group $RESOURCE_GROUP --name $BACKEND_NAME --yes

# Create new web app with Docker container support
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan ppanalyzer-plan \
  --name $BACKEND_NAME \
  --deployment-container-image-name $REGISTRY_NAME.azurecr.io/ppanalyzer-backend:latest

# Configure container registry credentials
REGISTRY_USERNAME=$(az acr credential show \
  --name $REGISTRY_NAME \
  --query "username" \
  --output tsv)

REGISTRY_PASSWORD=$(az acr credential show \
  --name $REGISTRY_NAME \
  --query "passwords[0].value" \
  --output tsv)

az webapp config container set \
  --name $BACKEND_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $REGISTRY_NAME.azurecr.io/ppanalyzer-backend:latest \
  --docker-registry-server-url https://$REGISTRY_NAME.azurecr.io \
  --docker-registry-server-user $REGISTRY_USERNAME \
  --docker-registry-server-password $REGISTRY_PASSWORD

# Enable container logging
az webapp log config \
  --name $BACKEND_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-container-logging filesystem

echo "✓ Web App configured for Docker"
```

---

### **Step 4: Add GitHub Secrets for Docker Deployment**

```bash
# Get registry credentials
REGISTRY_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --query "loginServer" --output tsv)
REGISTRY_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query "username" --output tsv)
REGISTRY_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" --output tsv)

# Get Azure credentials for GitHub Actions
az ad sp create-for-rbac \
  --name "ppanalyzer-github-actions" \
  --role contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP \
  --sdk-auth
```

**Copy the JSON output**, then add to GitHub:

1. Go to: [https://github.com/alktbihd/PPAnalyzer/settings/secrets/actions](https://github.com/alktbihd/PPAnalyzer/settings/secrets/actions)
2. Add these secrets:
   - `AZURE_CREDENTIALS`: (paste the JSON)
   - `REGISTRY_LOGIN_SERVER`: `ppanalyzerregistry.azurecr.io`
   - `REGISTRY_USERNAME`: (from command above)
   - `REGISTRY_PASSWORD`: (from command above)

---

### **Step 5: Add Environment Variables to Azure**

```bash
# Add via Azure Portal (don't paste keys in terminal!)
# Or use this method:

# Add app settings (use Portal for sensitive data!)
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $BACKEND_NAME \
  --settings \
    OPENAI_API_KEY="<add-via-portal>" \
    HEYGEN_API_KEY="<add-via-portal>" \
    WEBSITES_PORT=8000
```

**Better: Add via Portal** → Configuration → Application settings

---

### **Step 6: Commit Docker Files and Deploy**

```bash
cd /Users/saeedalketbi/Desktop/Senior\ design\ 2/project/ppaudit

# Pull latest changes
git pull origin main

# Add Docker files
git add backend/Dockerfile backend/.dockerignore docker-compose.yml .github/workflows/docker-deploy.yml

# Commit
git commit -m "Add Docker deployment configuration"

# Push to trigger deployment
git push origin main

# Watch deployment
open https://github.com/alktbihd/PPAnalyzer/actions
```

---

### **Step 7: Upload Model to Container**

After first deployment, upload model via Azure Portal:

1. Portal → **ppanalyzer-backend**
2. **SSH** or **Advanced Tools (Kudu)**
3. Navigate to `/app/model/privbert_final/`
4. Upload `model.safetensors` and `thresholds.npy`
5. Restart: `az webapp restart --resource-group ppanalyzer-rg --name ppanalyzer-backend`

---

## 📊 Deployment Flow

```
Git push → GitHub Actions
    ↓
Build Docker image
    ↓
Push to Azure Container Registry
    ↓
Deploy to App Service
    ↓
App runs from container ✅
```

---

**Creating all Docker files now...**

