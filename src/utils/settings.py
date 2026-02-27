"""
Common settings
"""
import os

_file_path = os.path.dirname(os.path.abspath(__file__))

# Provided that settings.py is located in the "project_root/src/utils" directory
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(_file_path))
STORAGE_PATH = os.path.join(PROJECT_ROOT_PATH, 'storage')
EXPERIMENTS_PATH = os.path.join(STORAGE_PATH, 'experiments')
NOTEBOOKS_PATH = os.path.join(PROJECT_ROOT_PATH, 'notebooks')