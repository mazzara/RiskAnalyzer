# src/logger_config.py  

import logging
from logging.handlers import RotatingFileHandler 
from src.config import LOG_FILE, LOGGER_NAME 

MAX_LOG_SIZE = 5 * 1024 * 1024    # 5MB 
BACKUP_COUNT = 5                  # Keep last 5 log files

# ---- Create Log Rotation Settings ----
file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding='utf-8'
)

# --- Set Log Format --- 
log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S" 

# --- Configure logging --- 
logging.basicConfig(
    level=logging.DEBUG,
    format="(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", 
    handlers=[file_handler, logging.StreamHandler()]    #saves and also prints to console
    )

# --- Get rootlogger ---
logger = logging.getLogger(LOGGER_NAME)
