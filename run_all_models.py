"""
Sequential training runner for all Residual Attention Network models.

This script runs all 8 model configurations sequentially (no parallel execution)
to avoid TensorFlow dataset loading conflicts.

Models trained:
- attention56_ARL, attention56_NAL
- attention92_ARL, attention92_NAL
- attention128_ARL, attention128_NAL
- attention164_ARL, attention164_NAL
"""

import subprocess
import sys
import os
import time
from datetime import datetime


# Python interpreter path (venv)
PYTHON_PATH = "/Users/ivy/Documents/semester_3/se03hw/bin/python"

# All model configurations to train
MODELS_TO_TRAIN = [
    ("attention56", "arl"),
    ("attention56", "nal"),
    ("attention92", "arl"),
    ("attention92", "nal"),
    ("attention128", "arl"),
    ("attention128", "nal"),
    ("attention164", "arl"),
    ("attention164", "nal"),
]


def get_model_id(model_name, att_type):
    """Get formatted model identifier."""
    return f"{model_name}_{att_type.upper()}"


def run_training_sequential(model_name, att_type, log_dir="logs", total_models=8, current_idx=1):
    """
    Run training for a single model configuration sequentially.
    
    Parameters:
        model_name (str): Model name (e.g., 'attention56')
        att_type (str): Attention type ('arl' or 'nal')
        log_dir (str): Directory for log files
        total_models (int): Total number of models to train
        current_idx (int): Current model index (for progress display)
    
    Returns:
        dict: Results dictionary with timing and status
    """
    model_id = get_model_id(model_name, att_type)
    
    # Create log directory if missing
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file path
    log_file_path = os.path.join(log_dir, f"{model_id}.log")
    
    # Build command
    cmd = [
        PYTHON_PATH,
        "train_cifar_tf.py",
        "--dataset", "cifar10",
        "--model", model_name,
        "--att-type", att_type,
    ]
    
    print(f"\n[{current_idx}/{total_models}] Running {model_id}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file_path}")
    print("-" * 60)
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open log file for writing
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Training started: {start_datetime}\n")
        log_file.write(f"Model: {model_id}\n")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()
        
        try:
            # Run subprocess and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                # Write to log file
                log_file.write(line)
                log_file.flush()
                # Also print to console (with prefix)
                print(f"[{model_id}] {line.rstrip()}")
            
            # Wait for process to complete
            exit_code = process.wait()
            
            # Record end time
            end_time = time.time()
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            training_time = end_time - start_time
            training_time_min = training_time / 60.0
            
            # Write completion info to log
            log_file.write("\n" + "=" * 60 + "\n")
            log_file.write(f"Training completed: {end_datetime}\n")
            log_file.write(f"Exit code: {exit_code}\n")
            log_file.write(f"Training time: {training_time_min:.2f} minutes ({training_time:.2f} seconds)\n")
            log_file.write(f"Status: {'SUCCESS' if exit_code == 0 else 'FAILED'}\n")
            
            success = exit_code == 0
            
            if success:
                print(f"\n✓ [{current_idx}/{total_models}] {model_id} completed successfully")
                print(f"  Time: {training_time_min:.2f} minutes")
            else:
                print(f"\n✗ [{current_idx}/{total_models}] {model_id} failed (exit code: {exit_code})")
                print(f"  Time: {training_time_min:.2f} minutes")
                print(f"  Check log: {log_file_path}")
            
            return {
                'model_id': model_id,
                'model_name': model_name,
                'att_type': att_type,
                'success': success,
                'exit_code': exit_code,
                'training_time_sec': training_time,
                'training_time_min': training_time_min,
                'log_file': log_file_path,
                'start_time': start_datetime,
                'end_time': end_datetime
            }
            
        except Exception as e:
            # Record end time even on exception
            end_time = time.time()
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            training_time = end_time - start_time
            training_time_min = training_time / 60.0
            
            error_msg = f"Error running {model_id}: {str(e)}"
            log_file.write("\n" + "=" * 60 + "\n")
            log_file.write(f"ERROR: {error_msg}\n")
            log_file.write(f"Training failed: {end_datetime}\n")
            log_file.write(f"Training time: {training_time_min:.2f} minutes\n")
            
            print(f"\n✗ [{current_idx}/{total_models}] {model_id} - ERROR: {e}")
            print(f"  Check log: {log_file_path}")
            
            return {
                'model_id': model_id,
                'model_name': model_name,
                'att_type': att_type,
                'success': False,
                'exit_code': -1,
                'training_time_sec': training_time,
                'training_time_min': training_time_min,
                'log_file': log_file_path,
                'start_time': start_datetime,
                'end_time': end_datetime,
                'error': str(e)
            }


def write_summary(results, summary_file="results/results_summary.txt"):
    """
    Write summary of all training runs to a file.
    
    Parameters:
        results (list): List of result dictionaries
        summary_file (str): Path to summary file
    """
    os.makedirs("results", exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Residual Attention Network - Training Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models: {len(results)}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<25} {'Time (min)':<15} {'Status':<10} {'Log File':<30}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            model_id = result['model_id']
            time_min = result['training_time_min']
            status = "OK" if result['success'] else "FAILED"
            log_file = result['log_file']
            
            f.write(f"{model_id:<25} {time_min:<15.2f} {status:<10} {log_file:<30}\n")
        
        f.write("-" * 70 + "\n")
        
        # Statistics
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_time = sum(r['training_time_min'] for r in results)
        
        f.write(f"\nSummary Statistics:\n")
        f.write(f"  Successful: {successful}/{len(results)}\n")
        f.write(f"  Failed: {failed}/{len(results)}\n")
        f.write(f"  Total time: {total_time:.2f} minutes ({total_time/60:.2f} hours)\n")
        
        if failed > 0:
            f.write(f"\nFailed models:\n")
            for result in results:
                if not result['success']:
                    f.write(f"  - {result['model_id']} (exit code: {result['exit_code']})\n")
                    if 'error' in result:
                        f.write(f"    Error: {result['error']}\n")


def print_final_summary(results):
    """
    Print a beautiful final summary table to console.
    
    Parameters:
        results (list): List of result dictionaries
    """
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<25} {'Time (min)':<15} {'Status':<10}")
    print("-" * 70)
    
    for result in results:
        model_id = result['model_id']
        time_min = result['training_time_min']
        status = "OK" if result['success'] else "FAILED"
        
        print(f"{model_id:<25} {time_min:<15.2f} {status:<10}")
    
    print("-" * 70)
    
    # Statistics
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_time = sum(r['training_time_min'] for r in results)
    
    print(f"\nTotal: {len(results)} models")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} minutes ({total_time/60:.2f} hours)")


def main():
    """Main function to orchestrate sequential training."""
    print("=" * 70)
    print("Residual Attention Network - Sequential Training Runner")
    print("=" * 70)
    print(f"\nPython interpreter: {PYTHON_PATH}")
    print(f"Total models to train: {len(MODELS_TO_TRAIN)}")
    print(f"Execution mode: Sequential (no parallel processing)")
    print(f"\nModels:")
    for i, (model_name, att_type) in enumerate(MODELS_TO_TRAIN, 1):
        print(f"  {i}. {get_model_id(model_name, att_type)}")
    
    # Check if Python path exists
    if not os.path.exists(PYTHON_PATH):
        print(f"\nERROR: Python interpreter not found at: {PYTHON_PATH}")
        print("Please update PYTHON_PATH in the script.")
        sys.exit(1)
    
    # Check if training script exists
    if not os.path.exists("train_cifar_tf.py"):
        print("\nERROR: train_cifar_tf.py not found in current directory.")
        sys.exit(1)
    
    print("\nStarting sequential training...")
    print("=" * 70)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Track overall start time
    overall_start_time = time.time()
    
    # Run all models sequentially
    results = []
    for idx, (model_name, att_type) in enumerate(MODELS_TO_TRAIN, 1):
        result = run_training_sequential(
            model_name, 
            att_type, 
            log_dir="logs",
            total_models=len(MODELS_TO_TRAIN),
            current_idx=idx
        )
        results.append(result)
        
        # Small delay between models to ensure clean state
        if idx < len(MODELS_TO_TRAIN):
            print("\nWaiting 5 seconds before next model...\n")
            time.sleep(5)
    
    # Calculate total time
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    # Write summary file
    write_summary(results)
    
    # Print final summary
    print_final_summary(results)
    
    print(f"\nTotal execution time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"Summary saved to: results/results_summary.txt")
    print("=" * 70)
    
    # Exit with error code if any model failed
    failed_count = sum(1 for r in results if not r['success'])
    if failed_count > 0:
        print(f"\nWarning: {failed_count} model(s) failed. Check logs/ directory for details.")
        sys.exit(1)
    else:
        print("\nAll models trained successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
