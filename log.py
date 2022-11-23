import logging
from logging.handlers import RotatingFileHandler
import copy
import sys

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message: str, use_color=True) -> str:
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            record_copy = copy.copy(record)  # make a copy of the record, file handler
            record_copy.levelname = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
        else:
            record_copy = record
        return logging.Formatter.format(self, record_copy)


class ColoredLogger(logging.Logger):
    FORMAT = "%(asctime)s - %(filename)s - %(levelname)-18s - %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    COLOR_FORMAT = formatter_message(FORMAT, True)
    USE_COLOR = True

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.INFO)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(color_formatter)
        console.addFilter(lambda record: record.levelno < logging.WARNING)
        self.addHandler(console)

        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(color_formatter)
        console.addFilter(lambda record: record.levelno >= logging.WARNING)
        self.addHandler(console)

    def set_color(self, color):
        """
        Set the color of the logger

        Parameters
        ----------
        color : bool
            True to enable color, False to disable color
        """
        self.USE_COLOR = color
        for handler in self.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(ColoredFormatter(self.COLOR_FORMAT, color))

    def set_format(self, format: str, Handler: logging.Handler = None):
        """
        Set the format of the logger

        Parameters
        ----------
        format : str
            The format to use
        """
        self.FORMAT = format
        self.COLOR_FORMAT = formatter_message(self.FORMAT, self.USE_COLOR)
        if Handler is None:
            for handler in self.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(ColoredFormatter(self.COLOR_FORMAT, self.USE_COLOR))
                else:
                    handler.setFormatter(logging.Formatter(self.FORMAT))
        else:
            Handler.setFormatter(logging.Formatter(self.FORMAT))

    def list_handlers(self):
        """
        List the handlers attached to the logger
        """
        for handler in self.handlers:
            print(handler)

    def print(self, *args, **kwargs):
        """
        Print a message to the logger

        Parameters
        ----------
        *args:
            The arguments to pass to the logger
        **kwargs:
            The keyword arguments to pass to the logger
            Available keywords:
                level: int
                    The logging level to use
                    Default: logging.INFO
                    Available levels: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
                sep: str
                    The separator to use when printing
                    Default: ' '
                end: str
                    The end character to use when printing
                    Default: ''
        """
        level = kwargs.pop('level', logging.INFO)
        sep = kwargs.pop('sep', ' ')
        end = kwargs.pop('end', '')
        kwargs.setdefault('stacklevel', 2)

        self.log(level, sep.join(map(str, args)) + end, **kwargs)

    def add_File_Handler(self, filename, maxBytes=1024 * 1024 * 10, backupCount=5, level=logging.INFO, formatter='%(asctime)s - %(filename)s - %(levelname)-8s - %(message)s (%(filename)s:%(lineno)d)'):
        """
        Add a file handler to the logger

        Parameters
        ----------
        filename : str
            The name of the file to log to
        maxBytes : int
            The maximum number of bytes to write to the file before rotating
            Default: 10MB
        backupCount : int
            The number of files to keep when rotating
            Default: 5
        level : int
            The logging level to use
            Default: logging.INFO
            Available levels: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        formatter : logging.Formatter
            The formatter to use for the file handler
            Default: logging.Formatter('%(asctime)s - %(filename)s - %(levelname)-8s - %(message)s (%(filename)s:%(lineno)d)')
        """
        filehandler = RotatingFileHandler(filename, maxBytes=maxBytes, backupCount=backupCount)
        filehandler.setLevel(level)
        filehandler.setFormatter(logging.Formatter(formatter))
        self.addHandler(filehandler)
        return


def setup_logger(name: str, color=True, filename=None, maxBytes=1024 * 1024 * 10, backupCount=5, level=logging.INFO, formatter='%(asctime)s - %(filename)s - %(levelname)-8s - %(message)s (%(filename)s:%(lineno)d)') -> ColoredLogger:
    """
    Setup a logger

    Parameters
    ----------
    name : str
        The name of the logger
        Recommended: __name__
    color : bool
        True to enable color, False to disable color
    filename : str
        The name of the file to log to
    maxBytes : int
        The maximum number of bytes to write to the file before rotating
        Default: 10MB
        Only used if filename is not None
    backupCount : int
        The number of files to keep when rotating
        Default: 5
        Only used if filename is not None
    level : int
        The logging level to use
        Default: logging.INFO
        Available levels: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    formatter : str
        The formatter to use for the file handler
        Default: '%(asctime)s - %(filename)s - %(levelname)-8s - %(message)s (%(filename)s:%(lineno)d)'

    Returns
    -------
    logger : logging.Logger
        The logger
    """
    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger(name)  # type: ColoredLogger
    logger.setLevel(level)
    logger.set_color(color)
    logger.set_format(formatter)
    if filename is not None:
        logger.add_File_Handler(filename, maxBytes=maxBytes, backupCount=backupCount, level=level, formatter=formatter)

    return logger


if __name__ == "__main__":
    logger = setup_logger(__name__, filename="test.log")

    logger.list_handlers()

    # logger.setLevel(logging.DEBUG)
    logger.info("Started")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    logger.print("This is a print message", "with multiple arguments", sep="\n")
    logger.print("This is a print message", "with multiple arguments", level=logging.WARNING)
