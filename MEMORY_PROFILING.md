# Memory Profiling Implementation Plan

This document outlines planned improvements to memory monitoring and profiling for the FOGGIE Dynamics analysis pipeline.

## Current State

The codebase currently uses:
- `psutil.Process().memory_info().rss` - Process resident set size (actual RAM usage)
- `tracemalloc.get_traced_memory()` - Python-level memory tracking (if enabled)
- Checkpoint logging via `log_memory_usage()` at specific stages

**Limitations:**
- Only shows memory at discrete checkpoints, not during operations
- Doesn't identify which specific allocations cause spikes
- No continuous monitoring to catch unexpected memory growth
- Limited detail on what's consuming memory

## Planned Improvements

### Phase 1: Enhanced Checkpoint Logging (Quick Win)

**Priority: HIGH** | **Effort: LOW**

Replace `log_memory_usage()` with a context manager approach for better granularity.

**Implementation:**
```python
@contextmanager
def track_memory(logger, operation: str):
    """Context manager to track memory usage during an operation."""
    process = psutil.Process(os.getpid())

    # Start measurements
    mem_before = process.memory_info().rss / 1024**3
    if tracemalloc.is_tracing():
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

    try:
        yield
    finally:
        # End measurements
        mem_after = process.memory_info().rss / 1024**3
        mem_delta = mem_after - mem_before

        logger.info(f"{operation}: {mem_delta:+.2f} GB (before: {mem_before:.2f} GB, after: {mem_after:.2f} GB)")

        if tracemalloc.is_tracing():
            snapshot_after = tracemalloc.take_snapshot()
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

            logger.info(f"Top 3 memory allocations for {operation}:")
            for stat in top_stats[:3]:
                logger.info(f"  {stat}")
```

**Usage:**
```python
# Instead of:
log_memory_usage(logger, "Load dataset")
ds = yt.load(snapshot_path)

# Use:
with track_memory(logger, "Load dataset"):
    ds = yt.load(snapshot_path)
```

**Benefits:**
- Shows memory delta (how much an operation consumed)
- Tracks peak memory during operation, not just after
- Provides allocation details via tracemalloc snapshots
- Drop-in replacement for existing checkpoints

**Tasks:**
- [ ] Add `track_memory()` context manager to `src/workflow.py`
- [ ] Replace all `log_memory_usage()` calls with `track_memory()` context manager
- [ ] Test on single halo run to verify output format

---

### Phase 2: Continuous Background Monitoring

**Priority: HIGH** | **Effort: MEDIUM**

Add continuous memory monitoring to catch unexpected spikes during long-running operations.

**Implementation:**
```python
class MemoryMonitor:
    """Background thread that continuously monitors memory usage."""

    def __init__(self, logger, interval=30):
        self.logger = logger
        self.interval = interval  # seconds
        self.running = False
        self.thread = None
        self.peak_memory = 0
        self.memory_history = []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        self.logger.info(f"Memory monitoring started (interval: {self.interval}s)")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.logger.info(f"Peak memory during run: {self.peak_memory:.2f} GB")

        # Log if there were significant spikes
        if len(self.memory_history) > 0:
            avg_memory = sum(self.memory_history) / len(self.memory_history)
            if self.peak_memory > avg_memory * 1.5:
                self.logger.warning(f"Memory spike detected: peak {self.peak_memory:.2f} GB vs avg {avg_memory:.2f} GB")

    def _monitor(self):
        process = psutil.Process(os.getpid())
        while self.running:
            mem_gb = process.memory_info().rss / 1024**3
            self.peak_memory = max(self.peak_memory, mem_gb)
            self.memory_history.append(mem_gb)
            time.sleep(self.interval)
```

**Module location:** Create new `src/monitoring.py` or add to `src/workflow.py`

**Integration points:**
- Start at beginning of `main()` in `code/foggie_analysis.py`
- Stop at end of script or in exception handler
- Add to SLURM job scripts for HPC runs

**Benefits:**
- Catches memory spikes that occur between checkpoints
- Tracks peak memory across entire run
- Detects memory leaks (growing trend over time)
- Minimal overhead (runs in background daemon thread)

**Tasks:**
- [ ] Create `MemoryMonitor` class in `src/monitoring.py` (or `src/workflow.py`)
- [ ] Add start/stop calls in `code/foggie_analysis.py` main function
- [ ] Integrate into SLURM job scripts (`job_scripts/*.sh`)
- [ ] Test on time-series run to verify background monitoring works

---

### Phase 3: Detailed Allocation Tracking

**Priority: MEDIUM** | **Effort: LOW**

Enhanced tracemalloc logging to identify specific memory-hungry allocations.

**Implementation:**
```python
def log_memory_details(logger, stage: str, top_n: int = 10):
    """Log detailed memory usage with top allocations."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    logger.info(f"Memory at {stage}: {mem_gb:.2f} GB")

    if tracemalloc.is_tracing():
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        logger.info(f"Top {top_n} memory allocations:")
        for i, stat in enumerate(top_stats[:top_n], 1):
            logger.info(f"  #{i}: {stat.filename}:{stat.lineno} - {stat.size / 1024**2:.1f} MB")

        # Total traced
        total = sum(stat.size for stat in top_stats)
        logger.info(f"Total traced: {total / 1024**3:.2f} GB")

        # Group by file
        by_file = {}
        for stat in top_stats:
            if stat.filename not in by_file:
                by_file[stat.filename] = 0
            by_file[stat.filename] += stat.size

        logger.info("Top 5 files by memory usage:")
        for i, (filename, size) in enumerate(sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:5], 1):
            logger.info(f"  #{i}: {filename} - {size / 1024**2:.1f} MB")
```

**Use at key stages:**
- After dataset loading
- After profile generation
- After MCMC fitting
- Before/after large array operations

**Benefits:**
- Identifies which files/lines allocate the most memory
- Helps pinpoint optimization targets
- Grouped statistics show module-level memory usage

**Tasks:**
- [ ] Add `log_memory_details()` function to `src/workflow.py`
- [ ] Call at 3-5 key stages (dataset load, profile generation, fitting, etc.)
- [ ] Ensure `tracemalloc.start()` is called early in script

---

### Phase 4: Profiling Tools for Deep Analysis

**Priority: MEDIUM** | **Effort: MEDIUM**

Add external profiling tools for one-time deep dives into memory bottlenecks.

#### fil-profiler (Recommended for Data Science)

**Best for:** Finding peak memory and allocations in numpy/yt/scientific code

**Installation:**
```bash
pip install filprofiler
```

**Usage:**
```bash
# Run analysis with fil-profiler
fil-run code/foggie_analysis.py --halo 8508 --component star --save_data

# Generates interactive HTML report in fil-result/ directory
```

**Output includes:**
- Peak memory usage timeline
- Memory allocation flame graph
- Which functions allocate memory (including C extensions)
- Memory by type (numpy arrays, Python objects, etc.)

#### memory_profiler (Line-by-Line Analysis)

**Best for:** Finding exactly which lines in a function consume memory

**Installation:**
```bash
pip install memory-profiler
```

**Usage:**
```python
# Add decorator to functions of interest
from memory_profiler import profile

@profile
def create_profiles(ds, component, config, ...):
    # Function code
    ...

@profile
def fit_component_models(...):
    # Function code
    ...
```

**Run:**
```bash
python -m memory_profiler code/foggie_analysis.py --halo 8508 --component star
```

**Output:** Line-by-line memory usage showing increments per line

**Tasks:**
- [ ] Install `filprofiler` and `memory-profiler` in environment
- [ ] Run `fil-profiler` on representative halo (8508 or 5036) with all components
- [ ] Analyze report to identify top 3 memory bottlenecks
- [ ] Add `@profile` decorators to 3-5 key functions for line-level analysis
- [ ] Document findings and optimization opportunities

---

### Phase 5: Documentation and Maintenance

**Priority: LOW** | **Effort: LOW**

**Tasks:**
- [ ] Add memory profiling dependencies to `requirements.txt`:
  - `memory-profiler`
  - `filprofiler`
- [ ] Create "Performance Monitoring" section in `CLAUDE.md` with:
  - How to use `track_memory()` context manager
  - How to interpret `MemoryMonitor` output
  - When to use fil-profiler vs memory_profiler
  - Common memory optimization patterns for yt/numpy code
- [ ] Add memory profiling examples to docstrings
- [ ] Update HPC job script templates with MemoryMonitor integration

---

## Expected Outcomes

After implementing these improvements, you should be able to:

1. **Identify memory bottlenecks** - Know which operations consume the most memory
2. **Detect memory leaks** - Catch unexpected memory growth in long-running jobs
3. **Optimize memory usage** - Target specific allocations for optimization
4. **Debug OOM errors** - Understand why jobs exceed SLURM memory limits (e.g., `#SBATCH --mem=160G`)
5. **Right-size resource requests** - Request appropriate memory for different job types

## SLURM Memory Allocation Context

Current memory allocations in job scripts:
- `dm_only_analysis.sh`: 160 GB
- Standard runs: Typically 128-160 GB

With better profiling, you can:
- Determine if 160 GB is over/under-provisioned
- Identify if certain halos need more memory
- Optimize code to reduce memory footprint
- Catch memory spikes before job termination

---

## Implementation Priority

**Week 1 (High Priority):**
1. Phase 1: Context manager replacement
2. Phase 2: MemoryMonitor class and integration

**Week 2-3 (Medium Priority):**
3. Phase 3: Detailed allocation tracking
4. Phase 4: Run fil-profiler diagnostic

**Ongoing (Low Priority):**
5. Phase 5: Documentation and maintenance

---

## References

- [memory_profiler documentation](https://pypi.org/project/memory-profiler/)
- [fil-profiler documentation](https://pythonspeed.com/fil/)
- [Python tracemalloc module](https://docs.python.org/3/library/tracemalloc.html)
- [psutil documentation](https://psutil.readthedocs.io/)
