import os
import logging

log_dir = "logs"
log_level = logging.INFO

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, "app.log")

logging.basicConfig(
    filename=log_filename,
    level=log_level,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logging.getLogger().addHandler(console_handler)

def log_info(message):
    logging.info(message)

def log_warning(message):
    logging.warning(message)

def log_error(message):
    logging.error(message)

def log_exception(message):
    logging.exception(message)
