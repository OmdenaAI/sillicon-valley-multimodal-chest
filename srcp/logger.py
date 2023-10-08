import logging
import os
from datetime import datetime

# Create a log file with a timestamp as the file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs directory if it does not exist
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Set the log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] [%(levelname)s] [%(name)s] [%(lineno)d] - %(message)s",
    level=logging.INFO
)