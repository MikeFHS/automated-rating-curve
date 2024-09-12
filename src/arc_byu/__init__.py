import logging
import sys

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.WARNING)
LOG.propagate = False
logging_format = logging.Formatter('%(asctime)s - %(levelname)s -%(message)s', "%Y-%m-%d %H:%M:%S")

_handler = logging.StreamHandler(sys.stdout)  # creates the handler
_handler.setLevel(logging.WARNING)  # sets the handler info
_handler.setFormatter(logging_format)
LOG.addHandler(_handler)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.WARNING)
_root_logger.addHandler(_handler)

__version__ = '0.1.0'