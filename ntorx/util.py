import logging
from sys import stdout
from os import environ

def config_logger(fd, reset=True):
    handler = logging.StreamHandler(fd)
    frmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(frmt)
    logger = logging.getLogger("")
    if reset:
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
