#!/bin/bash

# Fly.io deployment script for MVP Quality Leads
set -e

echo "🚀 Starting Fly.io deployment for MVP Quality Leads..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first:"
    echo "   curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if user is logged in
if ! flyctl auth whoami &> /dev/null; then
    echo "🔐 Please log in to Fly.io first:"
    flyctl auth login
fi

# Set environment variables
echo "📝 Setting up environment variables..."

# Check if SIRENE_TOKEN is set
if [ -z "$SIRENE_TOKEN" ]; then
    echo "⚠️  SIRENE_TOKEN not set. You'll need to set it as a secret:"
    echo "   flyctl secrets set SIRENE_TOKEN=your_token_here"
fi

# Deploy the application
echo "🚀 Deploying to Fly.io..."
flyctl deploy

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at: https://mvp-quality-leads.fly.dev"
echo "🔍 Check logs with: flyctl logs"
echo "📊 Monitor with: flyctl status"
