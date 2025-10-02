# Chyll Lead MVP - Enhanced ML Pipeline

A production-ready lead scoring MVP that uses machine learning to predict company success based on historical deal data and French Sirene registry information.

## 🚀 Features

- **ML Pipeline**: Similarity-based scoring with company embeddings
- **Sirene Integration**: French company registry API integration
- **Supabase Backend**: Edge Functions, Database, and Authentication
- **React Frontend**: Modern UI with TypeScript and Tailwind CSS
- **Production Ready**: CORS, error handling, and deployment configurations

## 🏗️ Architecture

```
Frontend (React + TypeScript) 
    ↓ HTTP requests
FastAPI (Python + ML Pipeline)
    ↓ (when deployed)
Supabase Edge Functions (Deno)
    ↓ 
Supabase Database (PostgreSQL)
```

## 📁 Project Structure

```
├── frontend/                 # React + TypeScript frontend
│   ├── src/
│   │   ├── pages/           # UploadHistory, Discover pages
│   │   ├── lib/             # API helpers, CSV utilities
│   │   └── App.tsx          # Main app with routing
│   └── package.json
├── services/
│   └── fastapi/             # Python ML service
│       ├── app-production.py # Production-ready API
│       ├── app-enhanced.py   # Full ML pipeline
│       └── requirements.txt
├── supabase/                # Supabase configuration
│   ├── functions/           # Edge Functions
│   └── migrations/          # Database migrations
└── test-simple.html        # Browser testing interface
```

## 🛠️ Tech Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for build tooling
- **Tailwind CSS** for styling
- **React Router** for navigation

### Backend
- **FastAPI** with Python 3.11
- **Pandas & NumPy** for data processing
- **Similarity-based ML** for company scoring
- **Sirene API** for French company data

### Infrastructure
- **Supabase** for database and Edge Functions
- **PostgreSQL** for data storage
- **Deno** for Edge Function runtime

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd chyll-lead-mvp
```

### 2. Start Supabase

```bash
supabase start
```

### 3. Start FastAPI Service

```bash
cd services/fastapi
pip install -r requirements.txt
uvicorn app-production:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Train model
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"dev-tenant","rows":[{"company_name":"TechCorp France","deal_status":"won","website":"techcorp.fr","ape":"6201Z","created_year":"2020"}]}'

# Discover companies
curl -X POST http://localhost:8000/discover \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"dev-tenant","filters":{"ape_codes":["6201Z"],"regions":["Île-de-France"]}}'
```

## 🧪 Testing

### Browser Testing
Open `http://localhost:3000/test-simple.html` to test the API endpoints in your browser.

### Frontend Testing
Navigate to `http://localhost:5173` to use the React frontend.

## 📊 API Endpoints

### `/health`
Returns service status and configuration.

### `/train`
Trains ML model with historical deal data.
```json
{
  "tenant_id": "dev-tenant",
  "rows": [
    {
      "company_name": "TechCorp France",
      "deal_status": "won",
      "website": "techcorp.fr",
      "ape": "6201Z",
      "created_year": "2020"
    }
  ]
}
```

### `/discover`
Discovers and scores companies based on filters.
```json
{
  "tenant_id": "dev-tenant",
  "filters": {
    "ape_codes": ["6201Z"],
    "regions": ["Île-de-France"],
    "age_buckets": ["0-5"],
    "headcount_buckets": ["1-10"]
  }
}
```

## 🔧 Configuration

### Environment Variables

#### FastAPI Service
- `SIRENE_MODE`: "demo" | "api" | "bulk"
- `SIRENE_TOKEN`: Your INSEE Sirene API token
- `SIRENE_BULK_PATH`: Path to local Sirene data files

#### Frontend
- `VITE_SUPABASE_FUNCTIONS_URL`: Supabase Edge Functions URL

### Supabase Configuration
- Project ID: `bbjmztjnmcoqwehmhgdu`
- Edge Functions deployed and ready

## 🚀 Production Deployment

### Deploy FastAPI to Railway
1. Connect GitHub repository to Railway
2. Set environment variables:
   - `SIRENE_MODE=api`
   - `SIRENE_TOKEN=your_token`
3. Deploy with start command: `uvicorn app-production:app --host 0.0.0.0 --port $PORT`

### Deploy Frontend to Vercel
1. Connect GitHub repository to Vercel
2. Set build command: `npm run build`
3. Set output directory: `dist`

### Update Supabase Secrets
```bash
supabase secrets set FASTAPI_URL=https://your-app.railway.app --project-ref bbjmztjnmcoqwehmhgdu
```

## 🔑 Getting Sirene API Token

1. Go to [api.insee.fr](https://api.insee.fr)
2. Create account and register for Sirene API access
3. Generate OAuth token
4. Set as `SIRENE_TOKEN` environment variable

## 📈 ML Pipeline Details

### Similarity Scoring
- **Company Name Matching**: Text similarity analysis
- **Domain Analysis**: Website pattern recognition
- **APE Code Matching**: Industry classification alignment
- **Historical Success**: Weighted scoring based on past wins

### Features Used
- Web footprint analysis
- Company age buckets
- Industry codes (APE)
- Geographic regions
- Historical deal outcomes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API health endpoint
2. Review the logs in your deployment platform
3. Test with the browser interface
4. Create an issue in this repository

---

**Built with ❤️ for lead scoring and company discovery**