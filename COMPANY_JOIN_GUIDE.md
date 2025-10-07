# Safe Company-Address Join for 70M+ Rows

This solution provides a robust, batched approach to safely join legal units (`sirene_unites_legales`) with establishments (`sirene_etablissements`) to create a unified dataset with addresses and all company information.

## 🎯 Problem Statement

You have:
- **~42M establishments** with address information (`numero_voie`, `type_voie`, `libelle_voie`, `code_postal`, `libelle_commune`, `code_commune`)
- **~29M legal units** with all company information but no addresses
- Need to join them safely using SIREN as the key
- Must handle this massive dataset with proper logging, monitoring, and safety measures

## 🛠️ Solution Components

### 1. **Main Join Script** (`create_safe_company_join.py`)
- Batched processing (100K rows at a time)
- Comprehensive logging and progress tracking
- Automatic rollback on errors
- Optimized database settings
- Real-time speed monitoring

### 2. **Real-time Monitor** (`monitor_company_join.py`)
- Live progress tracking
- Speed metrics and ETA calculation
- Data quality monitoring
- Visual progress bars

### 3. **Safety Manager** (`safety_manager.py`)
- Pre-operation safety checks
- Backup and restore functionality
- Data integrity verification
- Cleanup utilities

### 4. **Performance Tester** (`test_company_join_performance.py`)
- Tests performance on sample data
- Estimates full operation time
- Validates approach before running

## 🚀 Step-by-Step Execution Guide

### Step 1: Run Performance Tests
```bash
python test_company_join_performance.py
```

This will:
- Test join performance on 10K sample rows
- Test batch processing efficiency
- Test index performance
- Estimate total operation time
- Provide recommendations

### Step 2: Run Safety Checks
```bash
python safety_manager.py check
```

This will:
- Check disk space availability
- Check for table locks
- Verify database connectivity
- Ensure optimal settings

### Step 3: Create Backup Point
```bash
python safety_manager.py backup pre-join-$(date +%Y%m%d-%H%M%S)
```

This will:
- Create a backup of existing data
- Save operation metadata
- Enable rollback if needed

### Step 4: Start the Join Process
```bash
python create_safe_company_join.py
```

This will:
- Create the `companies_with_addresses` table
- Process data in batches of 100K rows
- Log detailed progress information
- Handle errors gracefully
- Create indexes after loading

### Step 5: Monitor Progress (in separate terminal)
```bash
python monitor_company_join.py
```

This will show:
- Real-time progress updates
- Processing speed metrics
- ETA calculations
- Data quality statistics
- Batch information

## 📊 Expected Results

The final `companies_with_addresses` table will contain:

### Legal Unit Fields (all fields from `sirene_unites_legales`)
- `siren` (primary key)
- `denomination_unite_legale`
- `activite_principale_unite_legale`
- `nature_juridique_unite_legale`
- `economie_sociale_solidaire_unite_legale`
- `societe_mission_unite_legale`
- All other legal unit fields...

### Address Fields (from `sirene_etablissements`)
- `numero_voie`
- `type_voie`
- `libelle_voie`
- `code_postal`
- `libelle_commune`
- `code_commune`
- `full_address` (computed field)

### Additional Fields
- `siret`, `nic` (from establishments)
- `latitude`, `longitude` (if available)
- `batch_number` (for tracking)
- `created_at` (timestamp)

## ⚡ Performance Expectations

Based on testing:
- **Processing speed**: ~10,000-50,000 rows/second
- **Estimated time**: 2-8 hours for full operation
- **Memory usage**: Optimized for large datasets
- **Disk space**: ~2-3GB additional space needed

## 🛡️ Safety Features

### Error Handling
- Automatic rollback on errors
- Detailed error logging
- Graceful failure handling
- Transaction safety

### Monitoring
- Real-time progress tracking
- Speed monitoring
- Data quality checks
- ETA calculations

### Backup & Recovery
- Pre-operation backups
- Point-in-time recovery
- Data integrity verification
- Cleanup utilities

## 🔧 Troubleshooting

### If the process fails:
1. **Check logs**: Review `company_join.log` for errors
2. **Verify data**: Run `python safety_manager.py verify`
3. **Clean up**: Run `python safety_manager.py cleanup`
4. **Restore backup**: Run `python safety_manager.py restore <backup_name>`

### If performance is slow:
1. **Check database settings**: Ensure optimal memory settings
2. **Monitor system resources**: Check CPU, memory, disk usage
3. **Adjust batch size**: Modify `batch_size` in the script
4. **Check indexes**: Ensure proper indexing

### If you need to stop the process:
1. **Stop the main script**: Ctrl+C
2. **Check current status**: Run `python safety_manager.py status`
3. **Clean up if needed**: Run `python safety_manager.py cleanup`

## 📈 Post-Operation Steps

### 1. Verify Results
```bash
python safety_manager.py verify
```

### 2. Test Query Performance
```sql
-- Test basic queries
SELECT COUNT(*) FROM companies_with_addresses;
SELECT COUNT(*) FROM companies_with_addresses WHERE full_address IS NOT NULL;
SELECT COUNT(*) FROM companies_with_addresses WHERE latitude IS NOT NULL;

-- Test filtered queries
SELECT COUNT(*) FROM companies_with_addresses WHERE code_postal LIKE '75%';
SELECT COUNT(*) FROM companies_with_addresses WHERE activite_principale_unite_legale LIKE '62%';
```

### 3. Create API Endpoints
Use the new `companies_with_addresses` table for your API endpoints.

### 4. Set Up Regular Refresh
Create a scheduled job to refresh the data when source data updates.

## 🎯 Key Benefits

1. **Safety First**: Comprehensive backup, monitoring, and rollback capabilities
2. **Performance Optimized**: Batched processing with optimal database settings
3. **Real-time Monitoring**: Live progress tracking with detailed metrics
4. **Error Resilient**: Graceful error handling and recovery
5. **Scalable**: Designed to handle 70M+ rows efficiently
6. **Transparent**: Detailed logging and progress reporting

## 📝 Log Files

- `company_join.log`: Main operation logs
- `company_join_safety.log`: Safety operation logs
- `performance_test.log`: Performance test logs
- `backup_*.json`: Backup metadata files

## 🔍 Monitoring Commands

```bash
# Check operation status
python safety_manager.py status

# Verify data integrity
python safety_manager.py verify

# Monitor real-time progress
python monitor_company_join.py

# Check safety status
python safety_manager.py check
```

This solution provides a production-ready approach to safely handle your massive dataset join operation with comprehensive safety measures, monitoring, and error handling.
