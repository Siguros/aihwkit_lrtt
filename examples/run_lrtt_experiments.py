#!/usr/bin/env python3
"""
Comprehensive LRTT experiment runner for parameter sweep.

Runs 125 different configurations:
- TRANSFER_EVERY: [2, 10, 100, 1000, 100000]  
- LRTT_RANKS: [[1,1], [4,4], [8,8], [16,16], [32,32]]
- LORA_ALPHA: [1, 4, 8, 16, 32]

Total: 5 × 5 × 5 = 125 experiments
Each experiment saves results to examples/results/ with parameter-based filename.
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json

# Experiment parameters
TRANSFER_EVERY_VALUES = [2, 10, 100, 1000, 100000]
LRTT_RANKS_VALUES = [[1, 1], [4, 4], [8, 8], [16, 16], [32, 32]]
LORA_ALPHA_VALUES = [1, 4, 8, 16, 32]

# Fixed parameters
EPOCHS = 50  # Set to 1 for quick testing, increase for full experiments
BATCH_SIZE = 64

def modify_lrtt_script(transfer_every, lrtt_ranks, lora_alpha, epochs=1):
    """Modify the LRTT script with specified parameters."""
    script_path = "03_mnist_training_lrtt.py"  # Script is in same directory
    
    # Read the file
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace parameter lines
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('EPOCHS = '):
            lines[i] = f'EPOCHS = {epochs}'
        elif line.startswith('LRTT_RANKS = '):
            lines[i] = f'LRTT_RANKS = {lrtt_ranks}  # Ranks for LRTT layers (input->hidden1, hidden1->hidden2)'
        elif line.startswith('TRANSFER_EVERY = '):
            lines[i] = f'TRANSFER_EVERY = {transfer_every} #fer A⊗B to C every N updates'
        elif line.startswith('LORA_ALPHA = '):
            lines[i] = f'LORA_ALPHA = {lora_alpha} #LoRA scaling factor - reduced from 100.0 for stability'
    
    # Write back the modified content
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines))

def run_single_experiment(exp_num, total_exp, transfer_every, lrtt_ranks, lora_alpha):
    """Run a single LRTT experiment with specified parameters."""
    
    print(f"\n{'='*80}")
    print(f"Experiment {exp_num}/{total_exp}")
    print(f"TRANSFER_EVERY={transfer_every}, RANKS={lrtt_ranks}, LORA_ALPHA={lora_alpha}")
    print(f"{'='*80}")
    
    # Expected output filename  
    expected_file = f"results/mnist_lrtt_results_te{transfer_every}_r{lrtt_ranks[0]}_{lrtt_ranks[1]}_a{lora_alpha}.0.xlsx"
    
    # Skip if file already exists
    if os.path.exists(expected_file):
        print(f"⏭️  Skipping: {expected_file} already exists")
        return {'status': 'skipped', 'reason': 'file_exists', 'file': expected_file}
    
    # Modify the script with current parameters
    modify_lrtt_script(transfer_every, lrtt_ranks, lora_alpha, EPOCHS)
    
    # Run the experiment
    start_time = time.time()
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'ml', 'python', '03_mnist_training_lrtt.py'
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout per experiment
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Experiment completed successfully in {elapsed_time:.1f}s")
            # Extract key metrics from output if possible
            lines = result.stdout.split('\\n')
            final_acc = "N/A"
            for line in lines:
                if "Final Test Accuracy" in line:
                    final_acc = line.split(':')[1].strip()
                    break
            
            return {
                'status': 'success',
                'time': elapsed_time,
                'final_accuracy': final_acc,
                'file': expected_file
            }
        else:
            print(f"❌ Experiment failed after {elapsed_time:.1f}s")
            print("Error output:")
            print(result.stderr[-500:])  # Last 500 characters
            return {
                'status': 'failed', 
                'time': elapsed_time,
                'error': result.stderr[-500:],
                'file': expected_file
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⏰ Experiment timed out after {elapsed_time:.1f}s")
        return {
            'status': 'timeout',
            'time': elapsed_time,
            'file': expected_file
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"💥 Experiment crashed: {e}")
        return {
            'status': 'crashed',
            'time': elapsed_time,
            'error': str(e),
            'file': expected_file
        }

def save_experiment_log(results, start_time):
    """Save experiment log to JSON file."""
    log_data = {
        'experiment_info': {
            'total_experiments': len(results),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration': (datetime.now() - start_time).total_seconds(),
            'parameters': {
                'transfer_every_values': TRANSFER_EVERY_VALUES,
                'lrtt_ranks_values': LRTT_RANKS_VALUES, 
                'lora_alpha_values': LORA_ALPHA_VALUES,
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE
            }
        },
        'results': results
    }
    
    # Save to results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    log_filename = f"lrtt_experiment_log_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    log_filepath = os.path.join(results_dir, log_filename)
    
    with open(log_filepath, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n📊 Experiment log saved to: {log_filepath}")
    return log_filepath

def print_summary(results):
    """Print experiment summary."""
    total = len(results)
    success = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'failed'])
    timeout = len([r for r in results if r['status'] == 'timeout'])
    skipped = len([r for r in results if r['status'] == 'skipped'])
    crashed = len([r for r in results if r['status'] == 'crashed'])
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total}")
    print(f"✅ Successful:     {success:3d} ({success/total*100:.1f}%)")
    print(f"❌ Failed:         {failed:3d} ({failed/total*100:.1f}%)")
    print(f"⏰ Timeout:        {timeout:3d} ({timeout/total*100:.1f}%)")
    print(f"⏭️  Skipped:        {skipped:3d} ({skipped/total*100:.1f}%)")
    print(f"💥 Crashed:        {crashed:3d} ({crashed/total*100:.1f}%)")
    
    if success > 0:
        total_time = sum([r['time'] for r in results if 'time' in r])
        avg_time = total_time / len([r for r in results if 'time' in r])
        print(f"\\nAverage time per experiment: {avg_time:.1f}s")
        print(f"Total experiment time: {total_time/3600:.1f} hours")
    
    # Show some successful experiments
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        print(f"\\n📁 Generated Excel files in examples/results/:")
        for i, result in enumerate(successful_results[:10]):  # Show first 10
            filename = os.path.basename(result['file'])
            print(f"  {i+1:2d}. {filename}")
        if len(successful_results) > 10:
            print(f"  ... and {len(successful_results)-10} more files")

def main():
    """Run all LRTT experiments."""
    print("🚀 Starting comprehensive LRTT parameter sweep")
    print(f"📋 Total experiments: {len(TRANSFER_EVERY_VALUES) * len(LRTT_RANKS_VALUES) * len(LORA_ALPHA_VALUES)}")
    print(f"⚙️  Configuration:")
    print(f"   TRANSFER_EVERY: {TRANSFER_EVERY_VALUES}")
    print(f"   LRTT_RANKS: {LRTT_RANKS_VALUES}")
    print(f"   LORA_ALPHA: {LORA_ALPHA_VALUES}")
    print(f"   EPOCHS: {EPOCHS}")
    print(f"   BATCH_SIZE: {BATCH_SIZE}")
    
    start_time = datetime.now()
    results = []
    exp_num = 0
    total_exp = len(TRANSFER_EVERY_VALUES) * len(LRTT_RANKS_VALUES) * len(LORA_ALPHA_VALUES)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run all combinations
    for transfer_every in TRANSFER_EVERY_VALUES:
        for lrtt_ranks in LRTT_RANKS_VALUES:
            for lora_alpha in LORA_ALPHA_VALUES:
                exp_num += 1
                
                result = run_single_experiment(exp_num, total_exp, transfer_every, lrtt_ranks, lora_alpha)
                result.update({
                    'experiment_number': exp_num,
                    'transfer_every': transfer_every,
                    'lrtt_ranks': lrtt_ranks,
                    'lora_alpha': lora_alpha
                })
                results.append(result)
                
                # Save intermediate log every 10 experiments
                if exp_num % 10 == 0:
                    save_experiment_log(results, start_time)
                    print_summary(results)
    
    # Final summary and log
    save_experiment_log(results, start_time)
    print_summary(results)
    
    print(f"\\n🎉 All experiments completed!")
    print(f"📊 Check results/ for all generated Excel files")
    print(f"📝 Detailed log saved with timestamps and parameters")

if __name__ == "__main__":
    main()