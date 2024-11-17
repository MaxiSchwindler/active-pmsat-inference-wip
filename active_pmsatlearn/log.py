import logging
import sys
import multiprocessing
import os

from pebble import concurrent

MAX_LEN_PROCESS_NAME = 24
DEBUG_EXT = 5

LOG_COLORS = {
    'DEBUG_EXT': 'cyan',
    'DEBUG': 'blue',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'magenta'
}


logging.addLevelName(DEBUG_EXT, 'DEBUG_EXT')


def debug_extended(self, message, *args, **kwargs):
    if self.isEnabledFor(DEBUG_EXT):
        self._log(DEBUG_EXT, message, args, **kwargs)


logging.Logger.debug_ext = debug_extended


class ColorFormatter(logging.Formatter):
    COLORS = {
        "cyan": "\033[96m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "magenta": "\033[95m",
        "reset": "\033[0m"
    }

    @staticmethod
    def is_git_bash():
        shell = os.getenv('SHELL')
        if shell is not None and 'bash.exe' in shell:
            return True
        return False

    @staticmethod
    def shell_supports_colors():
        return not ColorFormatter.is_git_bash()

    def format(self, record):
        record.message = record.getMessage()
        formatted_log = super().format(record)
        color_name = LOG_COLORS.get(record.levelname)
        color = self.COLORS.get(color_name, self.COLORS['blue'])
        if self.shell_supports_colors():
            return f"{color}{formatted_log}{self.COLORS['reset']}"
        else:
            return formatted_log


def _get_longest_logger_name():
    logger_dict = logging.root.manager.loggerDict
    longest_name = max(logger_dict.keys(), key=len, default="")
    return longest_name


def _get_longest_log_level():
    log_levels = list(LOG_COLORS.keys())
    longest_level = max(log_levels, key=len, default="")
    return longest_level


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        max_level_width = len(_get_longest_log_level())

        formatter = ColorFormatter(
            fmt=f"%(asctime)s %(levelname)-{max_level_width}s [%(processName)-{MAX_LEN_PROCESS_NAME}s]: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
    return logger


def set_current_process_name(name: str):
    if len(name) > MAX_LEN_PROCESS_NAME:
        name = name[:MAX_LEN_PROCESS_NAME]
        print(f"WARNING! Only process names up to {MAX_LEN_PROCESS_NAME} chars are allowed. Name is shortened to {name}")
    multiprocessing.current_process().name = name


@concurrent.process
def _test_pebble():
    set_current_process_name("test_pebble")
    logger = get_logger("test_pebble")
    logger.debug("debug message from pebble process")


if __name__ == "__main__":
    # Testing/Example usage
    logger = get_logger("longer_logger_name" )
    logger2 = get_logger("shorty")
    logger.info("This is an info message.")
    _test_pebble().result()
    logger2.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
