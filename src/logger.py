import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)


LOG_FILENAME = "../doc/logfile.log"
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def debug_log(cls, method, message):
    """Logs a message with level INFO on the root logger.
    Args:
        message (str): message describing some fact to save.
    """
    logging.debug("DEBUG cls:{} - {} - {}".format(cls, method, message))


def error_log(cls, method, message):
    """Logs a message with level INFO on the root logger.
    Args:
        message (str): message describing some fact to save.
    """
    logging.error("ERROR cls:{} - {} - {}".format(cls, method, message))
