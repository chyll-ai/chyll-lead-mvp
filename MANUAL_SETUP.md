# Manual Railway PostgreSQL Setup

## 🚨 **Current Issue:**
The Railway CLI is having issues with interactive prompts, so we need to set up PostgreSQL manually.

## 📋 **Manual Setup Steps:**

### **1. Open Railway Dashboard**
```bash
# Run this command to open your Railway dashboard
railway open
```

### **2. Add PostgreSQL Database**
In the Railway dashboard:
1. Go to your **exciting-purpose** project
2. Click **"+ New"** → **"Database"** → **"PostgreSQL"**
3. Wait for the database to be created

### **3. Get Connection Details**
Once PostgreSQL is created:
1. Click on the **PostgreSQL service**
2. Go to **"Variables"** tab
3. Copy these variables:
   - `PGHOST`
   - `PGPORT` 
   - `PGDATABASE`
   - `PGUSER`
   - `PGPASSWORD`

### **4. Set Environment Variables Locally**
```bash
# Replace with your actual values from Railway dashboard
export PGHOST="your-postgres-host.railway.internal"
export PGPORT="5432"
export PGDATABASE="railway"
export PGUSER="postgres"
export PGPASSWORD="your-password"
```

### **5. Test Connection**
```bash
python3 test_railway_connection.py
```

### **6. Start Loading Data**
```bash
python3 load_sirene_railway.py
```

## 🎯 **Alternative: Use Railway Web Interface**

If CLI continues to have issues:
1. **Open Railway Dashboard**: `railway open`
2. **Add PostgreSQL**: Click "+ New" → "Database" → "PostgreSQL"
3. **Get Variables**: Copy from Variables tab
4. **Set Locally**: Export the environment variables
5. **Run Scripts**: Use the local environment

## 🔧 **Quick Test Commands**

```bash
# Test connection
python3 test_railway_connection.py

# Start loading (this will take 20-30 minutes)
python3 load_sirene_railway.py

# Monitor progress
tail -f /dev/null  # Watch the logs
```

## 📊 **Expected Results**

With Railway Pro (32GB RAM):
- **Loading Speed**: ~1-2M rows/minute
- **Total Time**: ~20-30 minutes
- **Final Size**: ~42M establishments + ~20M legal units
- **Storage**: ~50GB total

Your Railway Pro setup is perfect for this! 🚀
