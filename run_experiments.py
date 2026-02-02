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
        
    # Set up environment with SUMO tools
    env = os.environ.copy()
    sumo_home = env.get("SUMO_HOME", "/usr/share/sumo")
    sumo_tools = os.path.join(sumo_home, "tools")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] += os.pathsep + sumo_tools
    else:
        env["PYTHONPATH"] = sumo_tools

    try:
        # 1. Simulate
        print(f"  > Simulating...")
        subprocess.run(
            [sys.executable, "main.py", "--config", temp_config_path, "--mode", "simulate"],
            check=True,
            env=env
        )
        
        # 2. Analyze
        print(f"  > Analyzing...")
        # Since main.py writes to detection_logs_<STEPS>.txt, it should handle the different step counts automatically
        subprocess.run(
            [sys.executable, "main.py", "--config", temp_config_path, "--mode", "analyze"],
            check=True,
            env=env
        )
        
        # 3. Rename/Move Images (Optional if main.py handles it, but let's ensure uniqueness)
        image_dir = f"image/{scenario_name}"
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.endswith(".png"):
                    print(f"    Generated: {os.path.join(image_dir, filename)}")
        
        # 4. Evaluate
        print(f"  > Evaluating and Generating Drawings...")
        gt_file = config_data.get("PROBABILITY_FILE")
        net_file = config_data.get("NET_FILE")
        if gt_file and os.path.exists(gt_file):
            pred_file = f"data/{scenario_name}/result_{steps}.json"
            eval_output = f"data/{scenario_name}/evaluation_{steps}.json"
            eval_plot = f"image/{scenario_name}/compare_{steps}.png"
            if os.path.exists(pred_file):
                eval_args = [
                    sys.executable, "evaluate_detections.py",
                    "--gt", gt_file,
                    "--pred", pred_file,
                    "--output", eval_output,
                    "--plot", eval_plot
                ]
                if net_file:
                    eval_args.extend(["--net", net_file])
                
                subprocess.run(eval_args, check=True, env=env)
                print(f"    Generated evaluation: {eval_output}")
                print(f"    Generated drawing: {eval_plot}")
            else:
                print(f"    ! Prediction file not found: {pred_file}")
        else:
            print(f"    ! Ground truth file not found: {gt_file}")
                    
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
