#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="algorithm/evaluation"

# Set number of parallel jobs
N_JOBS=1

# Get list of algorithms from config
# ALGORITHMS=(PC FCI CDNOD AcceleratedPC GES FGES XGES GRaSP 
#             DirectLiNGAM AcceleratedLiNGAM GOLEM 
#             NOTEARSLinear 
#             HITONMB BAMB IAMBnPC MBOR InterIAMB)
            # Hybrid CALM NOTEARSNonlinear CORL
# ALGORITHMS=(CDNOD AcceleratedPC AcceleratedLiNGAM FCI GES GOLEM GRaSP IAMBnPC InterIAMB MBOR NOTEARSLinear PC XGES)
ALGORITHMS=(Integrated)

# Create a temporary directory for parallel job logs
LOG_DIR="$SCRIPT_DIR/benchmark_logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"

# Function to run benchmark for a single algorithm with logging
run_benchmark() {
    local algo=$1
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/${algo}_${timestamp}.log"
    
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting benchmark for algorithm: $algo" | tee -a "$log_file"
    
    # Run the benchmark with CPU restrictions and redirect both stdout and stderr to the log file
    # Using taskset to restrict CPU usage to the last 8 cores
    CUDA_VISIBLE_DEVICES=7 taskset -c 120-127 python "$SCRIPT_DIR/benchmarking_integrated.py" &>> "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Benchmark for algorithm $algo completed successfully." | tee -a "$log_file"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Benchmark for algorithm $algo encountered an error." | tee -a "$log_file"
    fi
}

export -f run_benchmark
export LOG_DIR
export SCRIPT_DIR

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting all benchmarks..."

# Check if sequential flag is provided
if [ "$1" == "--sequential" ]; then
    echo "Running benchmarks sequentially..."
    for algo in "${ALGORITHMS[@]}"; do
        run_benchmark "$algo"
    done
else
    echo "Running benchmarks in parallel..."
    # Run benchmarks in parallel using GNU parallel
    parallel --jobs $N_JOBS run_benchmark ::: "${ALGORITHMS[@]}"
fi

echo "[$(date +"%Y-%m-%d %H:%M:%S")] All benchmarks completed. Logs available in $LOG_DIR/"
