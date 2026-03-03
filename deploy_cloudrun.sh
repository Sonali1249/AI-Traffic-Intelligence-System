#!/usr/bin/env bash
set -e

# Usage: ./deploy_cloudrun.sh <PROJECT_ID> <REGION> <SERVICE_NAME>
# Example: ./deploy_cloudrun.sh my-gcp-project europe-west1 traffic-streamlit

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <PROJECT_ID> <REGION> <SERVICE_NAME>"
  exit 1
fi

PROJECT_ID=$1
REGION=$2
SERVICE_NAME=$3

# Build Docker image and push to Artifact Registry
IMAGE_URI="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Build the container image

echo "Building Docker image..."

docker build -t $IMAGE_URI .

# Push the image to Google Container Registry (or Artifact Registry)

echo "Pushing image to $IMAGE_URI..."

gcloud auth configure-docker

docker push $IMAGE_URI

# Deploy to Cloud Run

echo "Deploying to Cloud Run..."

gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_URI \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=${GEMINI_API_KEY:-}

echo "Deployment complete. Access your service at the URL provided by gcloud."
