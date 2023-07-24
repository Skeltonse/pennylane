"""This file provides support for logging framework filters. For more information please see the official Python documentation on filters at https://docs.python.org/3/library/logging.html#filter"""
import logging
import os
from logging import Filter


# pylint: disable=too-few-public-methods
class LocalProcessFilter(Filter):
    """
    Filters logs not originating from the current executing Python process ID.
    """

    def __init__(self):
        super().__init__()
        self._pid = os.getpid()

    def filter(self, record):
        if record.process == self._pid:
            return True
        return False


# pylint: disable=too-few-public-methods
class DebugOnlyFilter(Filter):
    """
    Filters logs that are less verbose than the DEBUG level (CRITICAL, ERROR, WARN & INFO).
    """

    def filter(self, record):
        super().__init__()
        if record.levelno > logging.DEBUG:
            return False
        return True
