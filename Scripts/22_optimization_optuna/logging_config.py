import logging
import sys

def configure_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(processName)s - %(message)s'
    ))
    root.addHandler(handler)