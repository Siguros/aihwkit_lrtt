#!/usr/bin/env python3
"""
Monitor and analyze LRTT experiment results.
"""

import os
import json
import pandas as pd
from datetime import datetime
import glob

def list_experiment_logs():
    """List all experiment log files."""
    results_dir = "results"
    log_files = glob.glob(os.path.join(results_dir, "lrtt_experiment_log_*.json"))
    
    if not log_files:
        print("No experiment logs found.")
        return []
    
    print(f"Found {len(log_files)} experiment log(s):")
    for i, log_file in enumerate(log_files, 1):
        print(f"  {i}. {os.path.basename(log_file)}")
    
    return log_files

def analyze_experiment_log(log_file):
    """Analyze a specific experiment log."""
    print(f"\n📊 Analyzing: {os.path.basename(log_file)}")
    print("="*60)
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    # Basic info
    exp_info = data['experiment_info']
    results = data['results']
    
    print(f"Experiment started: {exp_info['start_time']}")
    print(f"Experiment ended: {exp_info['end_time']}")
    print(f"Total duration: {exp_info['total_duration']/3600:.2f} hours")
    print(f"Total experiments: {exp_info['total_experiments']}")
    
    # Status summary
    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nStatus Summary:")
    for status, count in status_counts.items():
        percentage = count / len(results) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    
    # Success analysis
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        print(f"\n✅ Successful Experiments Analysis:")
        times = [r['time'] for r in successful]
        print(f"  Average time: {sum(times)/len(times):.1f}s")
        print(f"  Min time: {min(times):.1f}s")
        print(f"  Max time: {max(times):.1f}s")
        
        # Parameter analysis for successful experiments
        print(f"\n📈 Parameter Distribution (Successful Only):")
        
        # Count by parameter
        transfer_counts = {}
        rank_counts = {}
        alpha_counts = {}
        
        for r in successful:
            te = r['transfer_every']
            ranks = tuple(r['lrtt_ranks'])
            alpha = r['lora_alpha']
            
            transfer_counts[te] = transfer_counts.get(te, 0) + 1
            rank_counts[ranks] = rank_counts.get(ranks, 0) + 1
            alpha_counts[alpha] = alpha_counts.get(alpha, 0) + 1
        
        print(f"  Transfer Every: {dict(sorted(transfer_counts.items()))}")
        print(f"  Ranks: {dict(sorted(rank_counts.items()))}")
        print(f"  LoRA Alpha: {dict(sorted(alpha_counts.items()))}")

def count_result_files():
    """Count generated Excel result files."""
    results_dir = "results"
    excel_files = glob.glob(os.path.join(results_dir, "mnist_lrtt_results_*.xlsx"))
    
    print(f"\n📁 Excel Result Files: {len(excel_files)}")
    if excel_files:
        print("Recent files:")
        # Sort by modification time, show newest first
        excel_files.sort(key=os.path.getmtime, reverse=True)
        for i, file in enumerate(excel_files[:10]):  # Show first 10
            filename = os.path.basename(file)
            size = os.path.getsize(file) / 1024  # Size in KB
            mtime = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  {i+1:2d}. {filename:<50} ({size:.1f}KB, {mtime.strftime('%H:%M:%S')})")
        
        if len(excel_files) > 10:
            print(f"  ... and {len(excel_files)-10} more files")

def create_results_summary():
    """Create a summary CSV of all successful experiments."""
    results_dir = "results"
    log_files = glob.glob(os.path.join(results_dir, "lrtt_experiment_log_*.json"))
    
    if not log_files:
        print("No experiment logs found for summary.")
        return
    
    # Use the most recent log
    latest_log = max(log_files, key=os.path.getmtime)
    
    with open(latest_log, 'r') as f:
        data = json.load(f)
    
    # Extract successful experiments
    successful = [r for r in data['results'] if r['status'] == 'success']
    
    if not successful:
        print("No successful experiments found.")
        return
    
    # Create summary DataFrame
    summary_data = []
    for result in successful:
        summary_data.append({
            'transfer_every': result['transfer_every'],
            'rank_layer1': result['lrtt_ranks'][0],
            'rank_layer2': result['lrtt_ranks'][1], 
            'lora_alpha': result['lora_alpha'],
            'experiment_time_s': result.get('time', 0),
            'final_accuracy': result.get('final_accuracy', 'N/A'),
            'excel_file': os.path.basename(result['file']),
            'status': result['status']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Sort by parameters for better organization
    df = df.sort_values(['transfer_every', 'rank_layer1', 'lora_alpha'])
    
    # Save summary
    summary_file = os.path.join(results_dir, "lrtt_experiments_summary.csv")
    df.to_csv(summary_file, index=False)
    
    print(f"\n📋 Experiment summary saved to: {summary_file}")
    print(f"   Total successful experiments: {len(df)}")
    print(f"   Parameter combinations tested: {len(df)}")
    
    return df

def main():
    """Main monitoring function."""
    print("🔍 LRTT Experiment Monitor")
    print("="*50)
    
    # List and analyze logs
    log_files = list_experiment_logs()
    
    if log_files:
        # Analyze the most recent log
        latest_log = max(log_files, key=os.path.getmtime)
        analyze_experiment_log(latest_log)
    
    # Count result files
    count_result_files()
    
    # Create summary
    try:
        create_results_summary()
    except Exception as e:
        print(f"Error creating summary: {e}")

if __name__ == "__main__":
    main()