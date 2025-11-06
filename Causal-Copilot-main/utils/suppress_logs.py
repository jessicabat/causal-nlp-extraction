"""
Suppress external library logs by monkey-patching their loggers
This should be imported before any other modules
"""

import logging
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Create a custom filter to suppress specific messages
class ExternalLibFilter(logging.Filter):
    def filter(self, record):
        # Suppress castle library messages
        if 'castle' in record.name.lower():
            return False
        if 'backend' in record.getMessage().lower():
            return False
        if 'pytorch' in record.getMessage().lower():
            return False
        return True

# Apply the filter to the root logger
root_logger = logging.getLogger()
root_logger.addFilter(ExternalLibFilter())

# Set external library loggers to ERROR level
external_libs = [
    'castle',
    'castle.backend', 
    'castle.algorithms',
    'httpx',
    'urllib3',
    'requests',
    'matplotlib',
    'sklearn',
    'numpy', 
    'pandas',
    'torch',
    'tensorflow',
    'transformers',
    'openai',
    'tigramite'
]

for lib in external_libs:
    logging.getLogger(lib).setLevel(logging.ERROR)
    logging.getLogger(lib).disabled = True