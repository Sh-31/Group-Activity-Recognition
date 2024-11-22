import os
import sys
import logging
from datetime import datetime
from IPython import get_ipython
import warnings

class Logger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.log_dir = os.path.join(exp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'training_{timestamp}.log')
        
        self.logger = self._setup_logger()
    
        self.log_file_handle = open(self.log_file, 'a', encoding='utf-8')
    
    def _setup_logger(self):
        """Set up the logger with both file and console output"""
        logger = logging.getLogger(f'kaggle_logger_{datetime.now().strftime("%H%M%S")}')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message):
        """Log info message using multiple approaches for redundancy"""
        try:
           
            self.logger.info(message)
     
            self.log_file_handle.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [INFO] {message}\n")
            self.log_file_handle.flush()
            
            sys.stdout.flush()
            
        except Exception as e:
            warnings.warn(f"Logging failed: {str(e)}. Attempting direct print.")
            print(f"DIRECT PRINT: {message}")
    
    def warning(self, message):
        self.logger.warning(message)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [WARNING] {message}")
    
    def error(self, message):
        self.logger.error(message)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [ERROR] {message}")
    
    def debug(self, message):
        self.logger.debug(message)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [DEBUG] {message}")
    
    def __del__(self):
        try:
            self.log_file_handle.close()
        except:
            pass

def setup_logging(exp_dir):
    """Create and return a Kaggle-compatible logger"""
    return Logger(exp_dir)


def force_display(message):
    """Force immediate display in Kaggle notebooks"""
    try:
        from IPython.display import display, HTML
        display(HTML(f"<pre>{message}</pre>"))
    except:
        print(message)
