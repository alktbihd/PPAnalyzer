#!/bin/bash
# Build Docker image locally with model files and push to Azure Container Registry

set -e

echo "🐳 Building Docker image with PrivBERT model..."

# Variables
RESOURCE_GROUP="ppanalyzer-rg"
REGISTRY_NAME="ppanalyzerregistry"  # Corrected ACR name
IMAGE_NAME="ppanalyzer-backend"
WEBAPP_NAME="ppanalyzer-backend"

# Get ACR login server
ACR_SERVER=$(az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query loginServer -o tsv)
echo "📦 ACR Server: $ACR_SERVER"

# Login to ACR
echo "🔐 Logging into Azure Container Registry..."
az acr login --name $REGISTRY_NAME

# Build image (includes model files from local filesystem)
echo "🏗️  Building Docker image (this may take 5-10 minutes)..."
cd backend
docker build --platform linux/amd64 -t $ACR_SERVER/$IMAGE_NAME:latest .
cd ..

# Push to ACR
echo "⬆️  Pushing image to ACR..."
docker push $ACR_SERVER/$IMAGE_NAME:latest

# Get image digest
IMAGE_DIGEST=$(az acr repository show --name $REGISTRY_NAME --image $IMAGE_NAME:latest --query digest -o tsv)
echo "✅ Image pushed: $ACR_SERVER/$IMAGE_NAME@$IMAGE_DIGEST"

# Update Web App to use new image
echo "🔄 Updating Web App to use new image..."
az webapp config container set \
  --name $WEBAPP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name "$ACR_SERVER/$IMAGE_NAME:latest"

# Restart Web App
echo "🔄 Restarting Web App..."
az webapp restart --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP

echo ""
echo "✅ Deployment complete!"
echo "🌐 Backend URL: https://$WEBAPP_NAME.azurewebsites.net"
echo "🔍 Check logs: az webapp log tail --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "Expected startup log:"
echo "  ✓ PrivBERT model loaded - using real classification"

