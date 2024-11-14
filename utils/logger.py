import os
import sys
import logging
from datetime import datetime


class TeeLogger:
    """Logger that writes to both file and console"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(exp_dir):
    """Set up logging to both file and console"""
    # Create logs directory
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)