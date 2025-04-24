"""
Provides functionalities to suppress various warnings by static type checker

Functions
---------
`unused_variables`    : A function to explicitly declare that variables are not used
"""


def unused_variables(*_) -> None:
    """
    Suppress `unused_variables` warnings by explicit declaration

    Parameters
    ----------
    An arbitrary number of variables that is intentionally unused
    """
    pass
