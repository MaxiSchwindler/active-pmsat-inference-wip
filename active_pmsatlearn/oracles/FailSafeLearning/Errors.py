###
# The code used in this file is copied from https://github.com/apferscher/ble-learning
#
# Following file/class has been copied:
# -- ble-learning/FailSafeLearning/Errors.py
#
###

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ConnectionError(Error):
    """
    Exception raised if connection to peripheral failed
    """

    def __init__(self):
        self.message = "Failed to detect advertisements of peripheral or connection failed"


class NonDeterministicError(Error):
    """
    Exception raised if non-deterministic behavior in query is detected
    """

    def __init__(self):
        self.message = "Non-determinism in query execution detected."

class TableError(Error):
    """
    Exception raised if non-deterministic behavior in query is detected
    """

    def __init__(self):
        self.message = "Observation Table needs to be updated."

class RepeatedNonDeterministicError(Error):
    """
    Exception raised if non-deterministic behavior in query is detected
    """
    def __init__(self):
        self.message = "Non-determinism in query execution detected."
