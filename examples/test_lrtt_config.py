#!/usr/bin/env python3
"""
Test script to verify LRTT configuration works before running full experiment sweep.
Tests 3 quick configurations with EPOCHS=1.
"""

import subprocess
import sys
import os

def test_configurations():
    """Test a few LRTT configurations to verify everything works."""
    
    # Test configurations
    test_configs = [
        {"transfer_every": 100, "ranks": [4, 4], "alpha": 4},   # Medium settings
        {"transfer_every": 10, "ranks": [1, 1], "alpha": 1},   # Small/fast settings  
        {"transfer_every": 1000, "ranks": [8, 8], "alpha": 8}, # Large settings
    ]
    
    print("🧪 Testing LRTT configurations before full experiment...")
    print(f"Testing {len(test_configs)} configurations with EPOCHS=1")
    
    success_count = 0
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n📋 Test {i}/{len(test_configs)}")
        print(f"   TRANSFER_EVERY={config['transfer_every']}")
        print(f"   RANKS={config['ranks']}")
        print(f"   LORA_ALPHA={config['alpha']}")
        
        # Modify the script
        modify_lrtt_script(config['transfer_every'], config['ranks'], config['alpha'], epochs=1)
        
        # Run the test
        try:
            result = subprocess.run([
                'conda', 'run', '-n', 'ml', 'python', '03_mnist_training_lrtt.py'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print("   ✅ SUCCESS")
                success_count += 1
                
                # Check for output file
                expected_file = f"results/mnist_lrtt_results_te{config['transfer_every']}_r{config['ranks'][0]}_{config['ranks'][1]}_a{config['alpha']}.0.xlsx"
                if os.path.exists(expected_file):
                    print(f"   ✅ Excel file created: {os.path.basename(expected_file)}")
                else:
                    print(f"   ⚠️  Excel file missing: {os.path.basename(expected_file)}")
            else:
                print("   ❌ FAILED")
                print(f"   Error: {result.stderr[-200:]}")  # Last 200 chars
                
        except subprocess.TimeoutExpired:
            print("   ⏰ TIMEOUT")
        except Exception as e:
            print(f"   💥 CRASH: {e}")
    
    print(f"\n📊 Test Summary:")
    print(f"   Successful: {success_count}/{len(test_configs)}")
    
    if success_count == len(test_configs):
        print("   🎉 All tests passed! Ready for full experiment sweep.")
        return True
    else:
        print("   ⚠️  Some tests failed. Check configuration before running full sweep.")
        return False

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

if __name__ == "__main__":
    success = test_configurations()
    
    if success:
        print(f"\n🚀 Ready to run full experiment sweep!")
        print(f"   Run: python examples/run_lrtt_experiments.py")
    else:
        print(f"\n🛑 Fix issues before running full experiment sweep.")