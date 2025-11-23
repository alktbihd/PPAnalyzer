#!/bin/bash

# Docker Deployment Script for Azure
# Run this script to deploy PPAnalyzer backend with Docker

set -e  # Exit on error

echo "======================================"
echo "PPAnalyzer Docker Deployment to Azure"
echo "======================================"

# Variables
RESOURCE_GROUP="ppanalyzer-rg"
LOCATION="uaenorth"
REGISTRY_NAME="ppanalyzerregistry"
BACKEND_NAME="ppanalyzer-backend"
PLAN_NAME="ppanalyzer-plan"

echo ""
echo "Step 1: Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --sku Basic \
  --location $LOCATION \
  --admin-enabled true

echo ""
echo "Step 2: Getting registry credentials..."
REGISTRY_LOGIN_SERVER=$(az acr show --name $REGISTRY_NAME --query "loginServer" --output tsv)
REGISTRY_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query "username" --output tsv)
REGISTRY_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" --output tsv)

echo "Registry: $REGISTRY_LOGIN_SERVER"

echo ""
echo "Step 3: Deleting old web app (if exists)..."
az webapp delete --resource-group $RESOURCE_GROUP --name $BACKEND_NAME --yes || true

echo ""
echo "Step 4: Creating web app with container support..."
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan $PLAN_NAME \
  --name $BACKEND_NAME \
  --deployment-container-image-name $REGISTRY_LOGIN_SERVER/ppanalyzer-backend:latest

echo ""
echo "Step 5: Configuring container settings..."
az webapp config container set \
  --name $BACKEND_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $REGISTRY_LOGIN_SERVER/ppanalyzer-backend:latest \
  --docker-registry-server-url https://$REGISTRY_LOGIN_SERVER \
  --docker-registry-server-user $REGISTRY_USERNAME \
  --docker-registry-server-password $REGISTRY_PASSWORD

echo ""
echo "Step 6: Enabling container logging..."
az webapp log config \
  --name $BACKEND_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-container-logging filesystem

echo ""
echo "Step 7: Configuring app settings..."
echo "⚠️  Please add your API keys via Azure Portal:"
echo "   1. Go to: https://portal.azure.com"
echo "   2. Find: ppanalyzer-backend"
echo "   3. Configuration → Application settings"
echo "   4. Add: OPENAI_API_KEY and HEYGEN_API_KEY"
echo ""
echo "Press Enter after you've added the keys..."
read

echo ""
echo "======================================"
echo "✓ Azure resources configured!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Add these secrets to GitHub:"
echo "   https://github.com/alktbihd/PPAnalyzer/settings/secrets/actions"
echo ""
echo "   REGISTRY_LOGIN_SERVER = $REGISTRY_LOGIN_SERVER"
echo "   REGISTRY_USERNAME = $REGISTRY_USERNAME"
echo "   REGISTRY_PASSWORD = [COPY FROM BELOW]"
echo ""
echo "Registry Password:"
echo "$REGISTRY_PASSWORD"
echo ""
echo "2. Get Azure credentials for GitHub:"
az ad sp create-for-rbac \
  --name "ppanalyzer-github-actions" \
  --role contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP \
  --sdk-auth

echo ""
echo "Copy the JSON above and add as secret: AZURE_CREDENTIALS"
echo ""
echo "3. Commit Docker files and push:"
echo "   git add ."
echo "   git commit -m 'Add Docker deployment'"
echo "   git push origin main"
echo ""
echo "4. Watch deployment:"
echo "   https://github.com/alktbihd/PPAnalyzer/actions"

