#!/usr/bin/env python3
"""
Script to run baseline experiments with different device configurations.
Automatically runs both IdealizedPreset and TikiTakaIdealizedPreset and saves separate Excel files.
"""

import subprocess
import sys
import os

def run_baseline_experiment(device_type):
    """Run baseline experiment with specified device type."""
    print(f"\n{'='*60}")
    print(f"Running baseline experiment with {device_type}")
    print(f"{'='*60}")
    
    # Modify the DEVICE_TYPE in the baseline script
    script_path = "03_mnist_training_baseline.py"
    
    # Read the file
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace the DEVICE_TYPE line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('DEVICE_TYPE = '):
            lines[i] = f'DEVICE_TYPE = "{device_type}"  # Options: "IdealizedPreset", "TikiTakaIdealizedPreset"'
            break
    
    # Write back the modified content
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines))
    
    # Run the experiment
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ {device_type} experiment completed successfully")
            print("Output:")
            print(result.stdout[-1000:])  # Last 1000 characters
        else:
            print(f"❌ {device_type} experiment failed")
            print("Error:")
            print(result.stderr[-1000:])
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {device_type} experiment timed out after 1 hour")
    except Exception as e:
        print(f"💥 {device_type} experiment crashed: {e}")

def main():
    """Run all baseline experiments."""
    print("Starting baseline experiments for MNIST training")
    print("This will generate Excel files for comparison with LRTT results")
    
    # List of device types to test
    device_types = [
        "IdealizedPreset",
        "TikiTakaIdealizedPreset"
    ]
    
    for device_type in device_types:
        run_baseline_experiment(device_type)
    
    print(f"\n{'='*60}")
    print("All baseline experiments completed!")
    print("Check the generated Excel files in results/:")
    
    results_dir = "results"
    
    # List generated Excel files
    for device_type in device_types:
        expected_file = os.path.join(results_dir, f"mnist_baseline_results_{device_type}_e30.xlsx")
        if os.path.exists(expected_file):
            print(f"  ✅ {expected_file}")
        else:
            print(f"  ❌ {expected_file} (not found)")
    
    print("\nCompare these baseline results with LRTT results:")
    lrtt_file = os.path.join(results_dir, "mnist_lrtt_results_te100_r16_16_a16.0.xlsx")
    if os.path.exists(lrtt_file):
        print(f"  ✅ {lrtt_file}")
    else:
        print(f"  ❌ {lrtt_file} (run LRTT experiment first)")

if __name__ == "__main__":
    main()