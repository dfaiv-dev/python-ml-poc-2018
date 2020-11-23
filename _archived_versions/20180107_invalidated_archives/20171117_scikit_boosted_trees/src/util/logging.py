import logging
import os
import sys


def setup_default(log_path: str = None, level=logging.DEBUG):
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    logging.root.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    logging.root.addHandler(stderr_handler)

    if log_path:
        file_handler = logging.FileHandler(os.path.join(log_path, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler)

    logging.root.setLevel(level)
