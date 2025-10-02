#!/bin/bash
# Deploy script for FastAPI backend

echo "🚀 Deploying Chyll FastAPI Backend..."

# Check if we're in the right directory
if [ ! -f "services/fastapi/app-production.py" ]; then
    echo "❌ Error: Please run this script from the project root"
    exit 1
fi

echo "✅ Found FastAPI application"

# Create a simple deployment package
mkdir -p deploy
cp -r services/fastapi/* deploy/
cp requirements.txt deploy/ 2>/dev/null || echo "requirements.txt not found"

echo "📦 Created deployment package"

echo "🎯 Next steps:"
echo "1. Go to Railway/Render/Heroku"
echo "2. Create new project"
echo "3. Upload the 'deploy' folder or connect to GitHub"
echo "4. Set environment variables:"
echo "   - SIRENE_MODE=api"
echo "   - SIRENE_TOKEN=your_token"
echo "5. Set start command: uvicorn app-production:app --host 0.0.0.0 --port \$PORT"

echo "✅ Deployment package ready in 'deploy' folder"
