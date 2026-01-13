# TORCS Competition Project

This repository contains the TORCS racing platform and the Gym driver code for the project.  
The main focus is on developing and testing the driver in `gym_torcs/torcs_jm_par.py`.

## Project Structure

```
torcs_project/
├─ torcs/                  # TORCS platform code (do not edit)
├─ gym_torcs/              # Gym wrapper and driver code
│  └─ torcs_jm_par.py      # Main driver code (edit this file)
├─ .gitignore
├─ README.md
```

## Getting Started

1. Clone the repository:
```
git clone https://github.com/bushh3/TORC-Competition.git
cd TORC-Competition
```

2. Install requirements (if any):
```
pip install -r requirements.txt
```

3. Run the driver:
```
python gym_torcs/torcs_jm_par.py
```

## Editing the Driver

Edit only:
```
gym_torcs/torcs_jm_par.py
```
Do not modify files in `torcs/`.

## Team Workflow

1. Pull latest changes: `git pull`  
2. Edit the driver locally  
3. Stage and commit changes:
```
git add gym_torcs/torcs_jm_par.py
git commit -m "Describe your changes"
```

4. Push changes:
```
git push
```

5. Teammates should always `git pull` to stay up-to-date.

## Notes

- Ignore temporary files and compiled binaries (`.gitignore`)  
- Use relative paths so the driver works on all machines

## License

*(Add license here if needed)*
