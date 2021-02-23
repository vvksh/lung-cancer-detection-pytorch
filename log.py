import logging


def setup_custom_logger(name):
    fmt = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

