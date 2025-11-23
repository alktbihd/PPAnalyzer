#!/bin/bash

# PPAnalyzer GPU Deployment Script for Azure Container Instances
# This script creates a GPU-enabled container instance for fast classification

set -e

# Configuration
RESOURCE_GROUP="ppanalyzer-rg"
ACR_NAME="ppanalyzer"
CONTAINER_NAME="ppanalyzer-gpu"
IMAGE_NAME="ppanalyzer-backend-gpu"
DNS_LABEL="ppanalyzer-gpu"
LOCATION="eastus"  # GPU available in: eastus, westus2, westeurope

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 PPAnalyzer GPU Deployment${NC}"
echo "================================"

# Get API keys from current App Service or use provided ones
echo -e "\n${YELLOW}📋 Setting up API keys...${NC}"

# Check if keys are provided as environment variables
if [ -z "$OPENAI_API_KEY" ] || [ -z "$HEYGEN_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  API keys not provided as environment variables${NC}"
    echo "Trying to fetch from App Service..."
    
    OPENAI_KEY=$(az webapp config appsettings list \
      --name ppanalyzer-backend \
      --resource-group $RESOURCE_GROUP \
      --query "[?name=='OPENAI_API_KEY'].value | [0]" -o tsv 2>/dev/null)

    HEYGEN_KEY=$(az webapp config appsettings list \
      --name ppanalyzer-backend \
      --resource-group $RESOURCE_GROUP \
      --query "[?name=='HEYGEN_API_KEY'].value | [0]" -o tsv 2>/dev/null)
    
    if [ -z "$OPENAI_KEY" ] || [ -z "$HEYGEN_KEY" ]; then
        echo -e "${YELLOW}⚠️  Could not fetch keys automatically${NC}"
        echo ""
        read -p "Enter OPENAI_API_KEY: " OPENAI_KEY
        read -p "Enter HEYGEN_API_KEY: " HEYGEN_KEY
    fi
else
    OPENAI_KEY=$OPENAI_API_KEY
    HEYGEN_KEY=$HEYGEN_API_KEY
    echo "✓ Using provided API keys"
fi

# Build and push GPU Docker image
echo -e "\n${BLUE}🐳 Building GPU Docker image...${NC}"
echo "This will take 5-10 minutes..."

cd backend

# Login to ACR
echo -e "${BLUE}🔐 Logging into Azure Container Registry...${NC}"
az acr login --name $ACR_NAME

# Build with GPU Dockerfile
docker build -f Dockerfile.gpu -t ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:latest .

echo -e "${BLUE}⬆️  Pushing image to ACR...${NC}"
docker push ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:latest

cd ..

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
ACR_SERVER="${ACR_NAME}.azurecr.io"

# Create Container Instance with GPU
echo -e "\n${BLUE}🎮 Creating GPU Container Instance...${NC}"
echo "Location: $LOCATION (GPU K80 available)"

az container create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --image ${ACR_SERVER}/${IMAGE_NAME}:latest \
  --registry-login-server $ACR_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --dns-name-label $DNS_LABEL \
  --ports 8000 \
  --gpu-resource count=1 sku=K80 \
  --cpu 6 \
  --memory 16 \
  --environment-variables \
    OPENAI_API_KEY="$OPENAI_KEY" \
    HEYGEN_API_KEY="$HEYGEN_KEY" \
  --location $LOCATION

# Get the FQDN
FQDN=$(az container show \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --query ipAddress.fqdn -o tsv)

echo -e "\n${GREEN}✅ GPU Deployment Complete!${NC}"
echo "================================"
echo -e "🌐 GPU Backend URL: ${GREEN}http://${FQDN}:8000${NC}"
echo -e "🔍 Health Check: ${GREEN}http://${FQDN}:8000/api/health${NC}"
echo ""
echo "⚡ Expected Performance:"
echo "  - Classification: 5-10 seconds (vs 2 minutes on CPU)"
echo "  - 10-20x faster!"
echo ""
echo "💰 Cost: ~\$0.95/hour when running"
echo ""
echo "📊 Manage Container:"
echo "  Start:  az container start --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP"
echo "  Stop:   az container stop --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP"
echo "  Logs:   az container logs --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP"
echo "  Delete: az container delete --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP --yes"
echo ""
echo "🔄 Update Frontend:"
echo "  Change VITE_API_URL to: http://${FQDN}:8000"

