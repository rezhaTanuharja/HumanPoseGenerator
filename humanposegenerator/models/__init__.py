"""
Provides NN models to generate human poses.

Modules
-------
`sequential`    : models consisting of components that run sequentially.
`submodels`     : primitive models or layers
"""

from . import sequential, submodels

__all__ = [
    "sequential",
    "submodels",
]
