# Railway PostgreSQL Setup Guide

## 🚀 **Railway Pro Setup Complete!**

With your Railway Pro plan, you have excellent resources for the Sirene database:
- **32 vCPU, 32GB RAM** - Perfect for large data processing
- **250GB ephemeral disk** - Plenty of space for 42M rows
- **Vertical autoscaling** - Will scale up during loading
- **Persistent volumes** - Data won't be lost on restarts

## 📋 **Next Steps:**

### **1. Get Railway PostgreSQL Connection Details**
```bash
# Link to your project (run this interactively)
railway link

# Get database variables
railway variables

# Look for these variables:
# PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
```

### **2. Set Environment Variables Locally**
```bash
# Copy the PostgreSQL connection details from Railway
export PGHOST="your-railway-host"
export PGPORT="5432"
export PGDATABASE="railway"
export PGUSER="postgres"
export PGPASSWORD="your-password"
```

### **3. Test Connection**
```bash
python test_railway_connection.py
```

### **4. Start Loading Sirene Data**
```bash
python load_sirene_railway.py
```

## 🎯 **Expected Performance:**

With Railway Pro specs:
- **Loading Speed**: ~1-2M rows/minute (much faster than local)
- **Memory**: 32GB RAM can handle large chunks
- **Storage**: 250GB is plenty for 42M rows (~50GB total)
- **Reliability**: No crashes like local Docker

## 📊 **Database Structure:**

The script will create two tables:
- **`sirene_etablissements`**: ~42M physical locations
- **`sirene_unites_legales`**: ~20M legal entities

## 🔧 **If You Need Help:**

1. **Connection Issues**: Run `railway variables` to get credentials
2. **Loading Errors**: Check Railway logs with `railway logs`
3. **Performance**: Monitor with `railway metrics`

## 🎉 **Benefits Over Local Supabase:**

✅ **No Docker crashes** - Railway handles infrastructure
✅ **Persistent storage** - Data survives restarts  
✅ **Better performance** - 32GB RAM vs local limits
✅ **Production ready** - Same database for dev/prod
✅ **Automatic scaling** - Handles load spikes

Your Railway Pro setup is perfect for this project! 🚀
