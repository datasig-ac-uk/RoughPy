# RoughPy JAX Benchmarks

This directory contains ASV (Airspeed Velocity) benchmarks for the RoughPy JAX implementation.

## Setup and Usage

### Prerequisites
Install ASV and dependencies from the repository root:
```bash
pip install -e ".[benchmarks]"
```

This installs:
- `asv>=0.6.0` - Airspeed Velocity benchmarking framework
- `jax>=0.4.20` and `jaxlib>=0.4.20` - JAX for accelerated computations
- `virtualenv>=20.0.0` - Environment isolation for ASV
- `psutil>=5.0.0` - Memory profiling support

### Running Benchmarks

#### Initial Setup
```bash
# Navigate to benchmarks directory
cd benchmarks

# Initialize ASV (first time only)
asv machine --yes

# Run benchmarks on current branch
asv run
```

#### Continuous Use
```bash
# From benchmarks directory:

# Run benchmarks for specific commits
asv run main^..HEAD

# Run specific benchmark categories
asv run --bench "FreeTensor"

# Generate HTML report
asv publish
asv preview
```

#### Development Testing
```bash
# From benchmarks directory:

# Quick test of benchmark validity
asv dev

# Profile specific benchmark
asv profile bench_free_tensor_jax.FreeTensorBenchmarks.time_ft_fma
```

## Environment Configuration

ASV is configured to use **virtualenv** (compatible with your current venv setup) rather than conda. Key settings:

- **Environment type**: `virtualenv` (automatically detected)
- **Python versions**: 3.12 (configurable in `asv.conf.json`)
- **JAX versions**: 0.8.0 (matrix testing support)
- **Isolation**: Each benchmark run uses a clean virtual environment

## Performance Regression Detection

The configuration includes regression detection with:
- **Factor threshold**: 2.0x slowdown triggers alert
- **Percentage threshold**: 20% slowdown triggers alert
- **Historical tracking**: Results stored in `.asv/results` directory
