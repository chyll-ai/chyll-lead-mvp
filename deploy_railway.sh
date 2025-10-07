#!/bin/bash
# Deploy script for Railway

echo "🚀 Deploying FastAPI backend to Railway..."

# Check if Railway CLI is available
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Login to Railway (this will open browser)
echo "🔐 Logging into Railway..."
railway login

# Link to existing project or create new one
echo "🔗 Linking to Railway project..."
railway link

# Deploy
echo "📦 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your API will be available at: https://your-app.railway.app"
