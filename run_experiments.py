import os
import subprocess
import json
import shutil
import time
import sys

# Configurations to test
# Format: (Config Name, SIM_STEPS override)
EXPERIMENTS = [
    # Scenario 1: Brussels Rural (using default or override)
    ("config/brussels_rural_config.json", 30),
    
    # Scenario 2: Graz A2 (30 steps for speed test)
    ("config/Graz_A2_config.json", 30),
    

    
    # You can add more variations here, e.g. different SIM_STEPS
    # ("config/Graz_A2_config.json", 1800),
]

def run_experiment(config_path, steps):
    print(f"--- Running Experiment: {config_path} with {steps} steps ---")
    
    # Load config to get SCENARIO_NAME
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    scenario_name = config_data.get("SCENARIO_NAME", "unknown")
    
    # create temporal config file with modified SIM_STEPS
    temp_config_path = config_path.replace(".json", f"_temp_{steps}.json")
    config_data["SIM_STEPS"] = steps
    with open(temp_config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
        
    try:
        # 1. Simulate
        print(f"  > Simulating...")
        subprocess.run(
            [sys.executable, "main.py", "--config", temp_config_path, "--mode", "simulate"],
            check=True
        )
        
        # 2. Analyze
        print(f"  > Analyzing...")
        # Since main.py writes to detection_logs_<STEPS>.txt, it should handle the different step counts automatically
        subprocess.run(
            [sys.executable, "main.py", "--config", temp_config_path, "--mode", "analyze"],
            check=True
        )
        
        # 3. Rename/Move Images (Optional if main.py handles it, but let's ensure uniqueness)
        image_dir = f"image/{scenario_name}"
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.endswith(".png"):
                    # Only rename if it doesn't already have the step count (main.py adds it, but let's be safe)
                    # Current output: occupancy_grid_<type><STEPS>.png
                    # We leave it as is, or we can move it to a central 'results' folder
                    print(f"    Generated: {os.path.join(image_dir, filename)}")
                    
    except subprocess.CalledProcessError as e:
        print(f"  ! Error during experiment: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    print("Starting Batch Simulations...")
    for config_file, steps in EXPERIMENTS:
        run_experiment(config_file, steps)
    print("Done.")

if __name__ == "__main__":
    main()
