# Performance Optimization Guide for delete_closed_establishments.py

## Problem
Your original script was struggling with a **9.38 GB CSV file** containing **42 million records**. The script loads the entire CSV into memory, which causes:
- High memory usage (potentially 10+ GB RAM)
- Slow processing (15-30 minutes)
- Risk of system crashes
- No progress tracking

## Solution: Three Optimized Versions

### 🥇 Ultra-Optimized Version (RECOMMENDED)
**File:** `delete_closed_establishments_ultra_optimized.py`

**Key Features:**
- **Parallel Processing**: Uses 4 worker threads
- **Memory Management**: Processes 50k rows at a time
- **Progress Tracking**: Real-time progress with ETA
- **Database Optimization**: Creates indexes automatically
- **Memory Monitoring**: Tracks peak memory usage
- **Batch Operations**: Processes database in 500-record batches

**Expected Performance:**
- ⏱️ **3-7 minutes** (vs 15-30 minutes original)
- 💾 **~1-2 GB RAM** (vs 10+ GB original)
- 🚀 **3-5x faster** than original

### 🥈 Optimized Version (FALLBACK)
**File:** `delete_closed_establishments_optimized.py`

**Key Features:**
- **Chunked Processing**: 100k rows at a time
- **Progress Tracking**: Shows processing progress
- **Batch Database Operations**: 1000-record batches
- **Memory Management**: Garbage collection

**Expected Performance:**
- ⏱️ **5-10 minutes** (vs 15-30 minutes original)
- 💾 **~2-4 GB RAM** (vs 10+ GB original)
- 🚀 **2-3x faster** than original

### 🥉 Original Version (NOT RECOMMENDED)
**File:** `delete_closed_establishments.py`

**Issues:**
- Loads entire CSV into memory
- No progress tracking
- High memory usage
- Slow processing

## Quick Start

### 1. Install Dependencies
```bash
chmod +x setup_performance_optimization.sh
./setup_performance_optimization.sh
```

### 2. Run Ultra-Optimized Version
```bash
python3 delete_closed_establishments_ultra_optimized.py
```

### 3. Benchmark Performance (Optional)
```bash
python3 benchmark_performance.py
```

## Performance Optimizations Explained

### 1. Chunked CSV Reading
```python
# Original: Loads entire file
df = pd.read_csv('data/sirene/StockEtablissement_utf8.csv')  # 9.38 GB!

# Optimized: Processes in chunks
for chunk in pd.read_csv(csv_path, chunksize=50000):  # 50k rows at a time
    process_chunk(chunk)
```

### 2. Memory Management
```python
# Force garbage collection
del chunk, closed_chunk
gc.collect()

# Monitor memory usage
memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
```

### 3. Parallel Processing
```python
# Process multiple chunks simultaneously
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
```

### 4. Database Optimization
```python
# Create indexes for faster queries
cursor.execute("""
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_with_addresses_siren 
    ON companies_with_addresses (siren)
""")
```

### 5. Progress Tracking
```python
# Real-time progress with ETA
progress = (processed / total) * 100
eta = (elapsed / progress * 100) - elapsed
logger.info(f"Progress: {progress:.1f}% | ETA: {eta/60:.1f}min")
```

## Configuration Options

You can adjust performance settings in the ultra-optimized version:

```python
@dataclass
class PerformanceConfig:
    chunk_size: int = 50000      # Rows per chunk
    batch_size: int = 500        # Database batch size
    max_workers: int = 4         # Parallel workers
    memory_limit_mb: int = 1024  # Memory limit per worker
```

## Expected Results

For your 42M record CSV file:

| Version | Time | Memory | Speed Improvement |
|---------|------|--------|-------------------|
| Original | 15-30 min | 10+ GB | Baseline |
| Optimized | 5-10 min | 2-4 GB | 2-3x faster |
| Ultra-Optimized | 3-7 min | 1-2 GB | 3-5x faster |

## Troubleshooting

### If script still seems slow:
1. **Check system resources**: Close other applications
2. **Use SSD storage**: Faster I/O performance
3. **Increase chunk size**: If you have more RAM
4. **Reduce workers**: If CPU is overloaded

### If memory issues persist:
1. **Reduce chunk_size**: From 50000 to 25000
2. **Reduce batch_size**: From 500 to 250
3. **Close other applications**: Free up system memory

## Files Created

- `delete_closed_establishments_optimized.py` - Optimized version
- `delete_closed_establishments_ultra_optimized.py` - Ultra-optimized version
- `benchmark_performance.py` - Performance testing
- `setup_performance_optimization.sh` - Setup script
- `PERFORMANCE_OPTIMIZATION_GUIDE.md` - This guide

## Next Steps

1. **Run the ultra-optimized version** for best performance
2. **Monitor the logs** for progress updates
3. **Check memory usage** during processing
4. **Use the benchmark script** to compare versions

The ultra-optimized version should handle your 9.38 GB CSV file efficiently without memory issues!
