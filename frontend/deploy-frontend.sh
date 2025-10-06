#!/bin/bash

# Frontend Deployment Script
# Usage: ./deploy-frontend.sh [preview|production]

set -e

echo "🚀 Starting Frontend Deployment..."

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Must run from frontend directory"
    echo "Usage: cd frontend && ./deploy-frontend.sh"
    exit 1
fi

# Get deployment type
DEPLOY_TYPE=${1:-production}

echo "📦 Installing dependencies..."
npm install

echo "🔨 Building frontend..."
if [ "$DEPLOY_TYPE" = "production" ]; then
    npm run build:prod
else
    npm run build
fi

echo "🚀 Deploying to Vercel..."
if [ "$DEPLOY_TYPE" = "production" ]; then
    vercel --prod --yes
else
    vercel --yes
fi

echo "✅ Deployment complete!"
echo "🌐 Check your deployment at: https://chyll-lead-mvp.vercel.app"

# Clear local cache
echo "🧹 Clearing local cache..."
rm -rf dist/
rm -rf node_modules/.vite/

echo "🎉 All done!"
