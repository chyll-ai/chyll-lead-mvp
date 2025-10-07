# ESS Mission Companies Map MVP

A map-based interface for visualizing ESS (Économie Sociale et Solidaire) and Mission-driven companies with QPV and ZRR zone data.

## Features

- **Interactive Map**: Visualize 37,000+ ESS and Mission companies across France
- **Zone Detection**: QPV (Quartiers Prioritaires de la Ville) and ZRR (Zone de Revitalisation Rurale) integration
- **Real-time Data**: Live data from PostgreSQL database
- **Modern UI**: Apple-inspired design with React/TypeScript frontend

## Tech Stack

- **Backend**: FastAPI + PostgreSQL + Railway
- **Frontend**: React + TypeScript + Vite + Leaflet + Tailwind CSS + Vercel
- **Data**: SIRENE database with geocoding and zone matching

## Setup

### Backend (Railway)

1. Set environment variables in Railway:
   ```
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   ```

2. Deploy with Railway CLI:
   ```bash
   railway up
   ```

### Frontend (Vercel)

1. Set environment variable in Vercel:
   ```
   VITE_API_URL=https://your-railway-app.railway.app
   ```

2. Deploy with Vercel CLI:
   ```bash
   vercel --prod
   ```

## Production URLs

- **API**: https://chyll-lead-mvp-production.up.railway.app
- **Frontend**: https://frontend-chyll.vercel.app

## API Endpoints

- `GET /` - API status
- `GET /companies` - Get companies with filters
- `GET /stats` - Get company statistics

## Data Sources

- SIRENE database (French company registry)
- QPV zones (Priority Urban Areas)
- ZRR zones (Rural Revitalization Zones)
- OpenStreetMap for geocoding

## License

MIT License