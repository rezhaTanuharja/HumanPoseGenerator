"""
Provides data processing pipelines.

Modules
-------
`diffuser`      : compute diffused human pose and remember how to backtrack.
`encoders`      : encode scalar values such as time.
`speedometer`   : compute conditional velocity field.
"""

from . import diffuser, encoders, velocimeter

__all__ = [
    "diffuser",
    "encoders",
    "velocimeter",
]
