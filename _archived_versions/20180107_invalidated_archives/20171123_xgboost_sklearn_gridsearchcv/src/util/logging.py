import logging
import os
import sys


def setup_default(log_path: str = None, level=logging.DEBUG):
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logging.root.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stdout_handler.setFormatter(formatter)
    logging.root.addHandler(stderr_handler)

    if log_path:
        file_handler = logging.FileHandler(os.path.join(log_path, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

    logging.root.setLevel(level)
