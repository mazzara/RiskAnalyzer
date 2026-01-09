# src/config.py  
import os
import json 

# ----------- Base Directories ----------- 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
LOG_DIR = os.path.join(BASE_DIR, 'logs') 


# ----------- Logger Configuration -----------
LOG_FILE = os.path.join(LOG_DIR, 'app.log') 
LOGGER_NAME = 'myRiskOne'
