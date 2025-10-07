#!/bin/bash

echo "🚀 Building and deploying frontend to Vercel..."

# Build the project
echo "📦 Building project..."
npm run build

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
npx vercel --prod --yes

echo "✅ Deployment complete!"
