#!/bin/bash

# Quick GPU Container Control Script

CONTAINER_NAME="ppanalyzer-gpu"
RESOURCE_GROUP="ppanalyzer-rg"

case "$1" in
  start)
    echo "🚀 Starting GPU container..."
    az container start --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP
    echo "✅ Container started!"
    echo "Wait 30-60 seconds for model to load, then check:"
    FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn -o tsv)
    echo "http://${FQDN}:8000/api/health"
    ;;
  
  stop)
    echo "⏸️  Stopping GPU container..."
    az container stop --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP
    echo "✅ Container stopped (not being charged)"
    ;;
  
  restart)
    echo "🔄 Restarting GPU container..."
    az container restart --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP
    echo "✅ Container restarted!"
    ;;
  
  logs)
    echo "📋 Container logs:"
    az container logs --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP --follow
    ;;
  
  status)
    echo "📊 Container status:"
    az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME \
      --query "{State:instanceView.state,GPU:'GPU K80',CPU:'6 cores',RAM:'16 GB',FQDN:ipAddress.fqdn}" -o table
    ;;
  
  url)
    FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn -o tsv)
    echo "🌐 Backend URL: http://${FQDN}:8000"
    echo "🔍 Health: http://${FQDN}:8000/api/health"
    ;;
  
  delete)
    echo "⚠️  Deleting GPU container..."
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      az container delete --name $CONTAINER_NAME --resource-group $RESOURCE_GROUP --yes
      echo "✅ Container deleted"
    else
      echo "❌ Cancelled"
    fi
    ;;
  
  *)
    echo "PPAnalyzer GPU Control"
    echo "====================="
    echo ""
    echo "Usage: ./gpu-control.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start    - Start the GPU container (begin charging)"
    echo "  stop     - Stop the GPU container (stop charging)"
    echo "  restart  - Restart the container"
    echo "  logs     - View container logs (Ctrl+C to exit)"
    echo "  status   - Show container status"
    echo "  url      - Show backend URL"
    echo "  delete   - Delete the container"
    echo ""
    echo "Cost: ~\$0.95/hour when running, \$0 when stopped"
    ;;
esac

