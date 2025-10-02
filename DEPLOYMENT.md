# 🚀 Deployment Options for MVP Quality Leads

## Quick Start (Recommended)

### Option 1: Railway (Easiest)
```bash
# Install Railway CLI
curl -fsSL https://railway.app/install.sh | sh

# Login and deploy
railway login
railway init
railway variables set SIRENE_TOKEN=your_token_here
railway up
```

### Option 2: Render (Great Free Tier)
1. Go to https://dashboard.render.com
2. Connect your GitHub repo
3. Create new Web Service
4. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
   - **Environment**: Python 3
   - **Plan**: Free

### Option 3: Heroku (Classic)
```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login and deploy
heroku login
heroku create your-app-name
heroku config:set SIRENE_TOKEN=your_token_here
git push heroku main
```

## Environment Variables Needed

Set these in your chosen platform:
- `SIRENE_TOKEN`: Your INSEE API token
- `SIRENE_MODE`: `api` (or `demo` for testing)
- `PORT`: `8000` (usually auto-set)

## Testing Your Deployment

Once deployed, test these endpoints:
- **Health**: `GET /health`
- **Train**: `POST /train` (with sample data)
- **Discover**: `POST /discover` (after training)

## Platform Comparison

| Platform | Free Tier | Ease | ML Support | Best For |
|----------|-----------|------|------------|----------|
| Railway  | ✅ Good   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Quick MVP |
| Render   | ✅ Excellent | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Production |
| Heroku   | ❌ Paid   | ⭐⭐⭐ | ⭐⭐⭐ | Enterprise |
| Fly.io   | ✅ Good   | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High Performance |

## Troubleshooting

### Common Issues:
1. **Memory errors**: Upgrade to paid plan for ML libraries
2. **Timeout**: Increase timeout settings
3. **Cold starts**: Use paid plans for better performance

### Debug Commands:
```bash
# Railway
railway logs
railway status

# Render
# Check dashboard logs

# Heroku
heroku logs --tail
heroku ps
```
