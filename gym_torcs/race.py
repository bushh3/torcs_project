import subprocess
import os
import time
import sys

try:
    import pyautogui
except ImportError:
    print("CRITICAL: You need pyautogui installed.")
    print("Run: pip3 install pyautogui")
    sys.exit(1)

# --- CONFIGURATION ---
TORCS_DIR = "/Users/harrybush/Desktop/torcs_project/torcs"
GYM_TORCS_DIR = "/Users/harrybush/Desktop/torcs_project/gym_torcs"
PYTHON_CLIENT_SCRIPT = "torcs_jm_par.py"
WINE_CMD = "wine"

# --- COORDINATES ---
# Hardcoded based on your specific screen setup
CLICK_X = 330
CLICK_Y = 1532

def kill_zombies():
    print("--- Cleaning up old Wine processes ---")
    subprocess.run(["killall", "-9", "wineserver"], stderr=subprocess.DEVNULL)
    subprocess.run(["killall", "-9", "wine"], stderr=subprocess.DEVNULL)
    subprocess.run(["killall", "-9", "wtorcs.exe"], stderr=subprocess.DEVNULL)
    time.sleep(1)

def start_torcs():
    print(f"--- Launching TORCS ---")
    try:
        subprocess.Popen(
            [WINE_CMD, "wtorcs.exe"], 
            cwd=TORCS_DIR,
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("Error: 'wine' command not found.")
        sys.exit(1)

def brute_force_menus():
    print("--- Waiting 12 seconds for TORCS to load ---")
    time.sleep(12)
    
    print("--- AUTOMATION STARTING ---")
    print(f"Clicking specific coordinates: ({CLICK_X}, {CLICK_Y})")
    
    # 1. FORCE FOCUS CLICK (Using your coordinates)
    pyautogui.click(CLICK_X, CLICK_Y)
    time.sleep(0.5)
    
    # 2. PRESS ENTER REPEATEDLY
    # Pressing 5 times ensures we get through Race -> Quick Race -> New Race -> Start
    keys = ['enter', 'enter', 'enter', 'enter', 'enter']
    
    for k in keys:
        print(f"Pressing {k}...")
        pyautogui.press(k)
        time.sleep(1.5) # Wait for menu animation
        
    print("--- Menu Navigation Complete ---")
    print("--- Waiting 10 seconds for track to load ---")
    time.sleep(10)

def start_python_client():
    print(f"--- Launching Python Client ---")
    try:
        subprocess.Popen(
            ["python3", PYTHON_CLIENT_SCRIPT],
            cwd=GYM_TORCS_DIR
        )
        print("--- Client Connected ---")
    except Exception as e:
        print(f"Error starting python client: {e}")

if __name__ == "__main__":
    kill_zombies()
    start_torcs()
    brute_force_menus()
    start_python_client()