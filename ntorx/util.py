import logging
from sys import stdout
from os import environ, get_terminal_size

def config_logger(fd, level=logging.INFO, reset=True):
    handler = logging.StreamHandler(fd)
    frmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(frmt)
    logger = logging.getLogger("")
    if reset:
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger

