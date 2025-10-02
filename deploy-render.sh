#!/bin/bash

# Render deployment script for MVP Quality Leads
set -e

echo "🚀 Starting Render deployment for MVP Quality Leads..."

# Check if Render CLI is installed
if ! command -v render &> /dev/null; then
    echo "❌ Render CLI is not installed. Installing..."
    curl -fsSL https://cli.render.com/install | sh
    echo "✅ Render CLI installed!"
fi

# Check if user is logged in
if ! render auth whoami &> /dev/null; then
    echo "🔐 Please log in to Render first:"
    render auth login
fi

echo "📝 To deploy to Render:"
echo "1. Go to https://dashboard.render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Use these settings:"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: python run.py"
echo "   - Environment: Python 3"
echo "   - Plan: Free"
echo ""
echo "5. Add environment variables:"
echo "   - SIRENE_TOKEN: your_token_here"
echo "   - SIRENE_MODE: api"
echo "   - PORT: 8000"
echo ""
echo "6. Deploy!"

echo "✅ Render setup complete!"
echo "🌐 Your app will be available at: https://your-app-name.onrender.com"
