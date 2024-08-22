import logging
import os
from datetime import datetime

# creates logs folder
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# creates folder for all the 
LOG_FOLDER= f"{datetime.now().strftime('%m_%d_%Y')}"
logs_folder_path = os.path.join(logs_path, LOG_FOLDER)
os.makedirs(logs_folder_path, exist_ok=True)

LOG_FILE= f"{datetime.now().strftime('%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_folder_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")