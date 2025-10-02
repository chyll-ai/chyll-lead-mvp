#!/bin/bash

# Railway deployment script for MVP Quality Leads
set -e

echo "🚀 Starting Railway deployment for MVP Quality Leads..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed. Installing..."
    curl -fsSL https://railway.app/install.sh | sh
    echo "✅ Railway CLI installed!"
    echo "Please restart your terminal or run: source ~/.bashrc"
    exit 1
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log in to Railway first:"
    echo "Run: railway login"
    exit 1
fi

echo "📝 Setting up Railway deployment..."

# Set environment variables
echo "Setting SIRENE_TOKEN..."
railway variables --set 'SIRENE_TOKEN=a39ed0ac-2e48-4e0f-9ed0-ac2e48ce0ff1'

# Deploy the application
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your app will be available at the Railway URL!"
echo "🔍 Check logs with: railway logs"
echo "📊 Monitor with: railway status"
