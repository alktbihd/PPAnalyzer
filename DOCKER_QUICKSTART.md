# Docker Deployment - Quick Start

Simple step-by-step guide to deploy PPAnalyzer with Docker.

---

## 🚀 Option 1: Automated Script (Easiest)

Run the automated script:

```bash
cd /Users/saeedalketbi/Desktop/Senior\ design\ 2/project/ppaudit

# Run deployment script
./deploy-docker-azure.sh
```

Follow the prompts and add secrets to GitHub as instructed.

---

## 🔧 Option 2: Manual Steps

### **Step 1: Create Azure Container Registry**

```bash
RESOURCE_GROUP="ppanalyzer-rg"
REGISTRY_NAME="ppanalyzerregistry"
LOCATION="uaenorth"

az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --location $LOCATION --admin-enabled true
```

---

### **Step 2: Get Registry Credentials**

```bash
az acr credential show --name $REGISTRY_NAME
```

Copy the **username** and **password**.

---

### **Step 3: Update Web App for Docker**

```bash
BACKEND_NAME="ppanalyzer-backend"
REGISTRY_LOGIN_SERVER="$REGISTRY_NAME.azurecr.io"

# Delete old app
az webapp delete --resource-group $RESOURCE_GROUP --name $BACKEND_NAME --yes

# Create with Docker
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan ppanalyzer-plan \
  --name $BACKEND_NAME \
  --deployment-container-image-name $REGISTRY_LOGIN_SERVER/ppanalyzer-backend:latest
```

---

### **Step 4: Configure Container Registry**

```bash
REGISTRY_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query "username" --output tsv)
REGISTRY_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" --output tsv)

az webapp config container set \
  --name $BACKEND_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $REGISTRY_LOGIN_SERVER/ppanalyzer-backend:latest \
  --docker-registry-server-url https://$REGISTRY_LOGIN_SERVER \
  --docker-registry-server-user $REGISTRY_USERNAME \
  --docker-registry-server-password $REGISTRY_PASSWORD
```

---

### **Step 5: Add GitHub Secrets**

Go to: [https://github.com/alktbihd/PPAnalyzer/settings/secrets/actions](https://github.com/alktbihd/PPAnalyzer/settings/secrets/actions)

Add these 4 secrets:

1. **REGISTRY_LOGIN_SERVER**
   - Value: `ppanalyzerregistry.azurecr.io`

2. **REGISTRY_USERNAME**
   - Value: (from Step 2)

3. **REGISTRY_PASSWORD**
   - Value: (from Step 2)

4. **AZURE_CREDENTIALS**
   - Get with: 
   ```bash
   az ad sp create-for-rbac --name "ppanalyzer-github" --role contributor --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/ppanalyzer-rg --sdk-auth
   ```
   - Value: (paste the JSON output)

---

### **Step 6: Push to GitHub**

```bash
cd /Users/saeedalketbi/Desktop/Senior\ design\ 2/project/ppaudit

git add .
git commit -m "Add Docker deployment"
git push origin main
```

GitHub Actions will:
- Build Docker image
- Push to Azure Container Registry
- Deploy to App Service

---

### **Step 7: Upload Model**

Via Azure Portal:
1. Go to App Service
2. SSH or Kudu Console
3. Navigate to `/app/model/privbert_final/`
4. Upload `model.safetensors` (500MB)

---

## ✅ Verification

```bash
# Check if container is running
az webapp show --resource-group ppanalyzer-rg --name ppanalyzer-backend --query "state"

# Test endpoint
curl https://ppanalyzer-backend.azurewebsites.net/api/health

# View logs
az webapp log tail --resource-group ppanalyzer-rg --name ppanalyzer-backend
```

---

## 🎯 Quick Commands Summary

```bash
# 1. Run automated script
./deploy-docker-azure.sh

# 2. Add secrets to GitHub (follow script output)

# 3. Push to deploy
git add .
git commit -m "Add Docker deployment"
git push origin main

# 4. Watch deployment
open https://github.com/alktbihd/PPAnalyzer/actions
```

Done! 🎉

