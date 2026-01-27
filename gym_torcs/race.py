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
RESTART_X = 260
RESTART_Y = 490


def kill_zombies():
    print("--- Cleaning up old Wine processes ---")
    subprocess.run(["killall", "-9", "wineserver"], stderr=subprocess.DEVNULL)
    subprocess.run(["killall", "-9", "wine"], stderr=subprocess.DEVNULL)
    subprocess.run(["killall", "-9", "wtorcs.exe"], stderr=subprocess.DEVNULL)
    time.sleep(1)


def start_torcs():
    print("--- Launching TORCS ---")
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
    print(f"Clicking focus coordinates: ({CLICK_X}, {CLICK_Y})")

    # Force focus
    pyautogui.click(CLICK_X, CLICK_Y)
    time.sleep(0.5)

    # Navigate menus
    for _ in range(5):
        pyautogui.press("enter")
        time.sleep(1.5)

    print("--- Menu Navigation Complete ---")
    print("--- Waiting 10 seconds for track to load ---")
    time.sleep(10)


def start_python_client():
    print("--- Launching Python Client ---")
    try:
        proc = subprocess.Popen(
            ["python3", PYTHON_CLIENT_SCRIPT],
            cwd=GYM_TORCS_DIR
        )
        print("--- Client Connected ---")
        return proc
    except Exception as e:
        print(f"Error starting python client: {e}")
        return None


def auto_restart_race(client_process):
    print("--- Waiting for Python client shutdown ---")
    client_process.wait()

    print("--- Client shutdown detected ---")
    print("--- Restarting race sequence ---")
    time.sleep(2)

    # Step 1: restart button
    print(f"Clicking restart button at ({RESTART_X}, {RESTART_Y})")
    pyautogui.click(RESTART_X, RESTART_Y)
    time.sleep(1)

    # Step 2–4: click race start button three times
    for i in range(3):
        print(f"Clicking race button {i + 1}/3 at ({CLICK_X}, {CLICK_Y})")
        pyautogui.click(CLICK_X, CLICK_Y)
        time.sleep(1.5)

    print("--- Race restart sequence complete ---")


if __name__ == "__main__":
    kill_zombies()
    start_torcs()
    brute_force_menus()

    client_proc = start_python_client()
    if client_proc is not None:
        auto_restart_race(client_proc)