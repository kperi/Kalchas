#!/bin/bash

# Development script for Logios Flask app with auto-reload
echo "Starting Logios in development mode with auto-reload..."

# Stop any existing containers
docker compose down

# Build and start services with development overrides
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up --build

echo "Development server started. Flask app will auto-reload on file changes."
echo "Access the Flask UI at: http://localhost:5001"
echo "Access the Streamlit UI at: http://localhost:8502"
echo "OCR API at: http://localhost:8000"
echo "Document Layout API at: http://localhost:8602"