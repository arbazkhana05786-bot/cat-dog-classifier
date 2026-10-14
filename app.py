import os
import sys
from pathlib import Path

# Change working directory to the starter project folder and run app.py
PROJECT_DIR = Path(__file__).parent / "CatDog_CNN_Safe_Starter"
if PROJECT_DIR.exists():
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))
    import runpy
    runpy.run_path(str(PROJECT_DIR / "app.py"), run_name="__main__")
else:
    print(f"Error: Project directory {PROJECT_DIR} not found.")
